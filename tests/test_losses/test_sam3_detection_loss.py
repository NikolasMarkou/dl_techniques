"""Tests for the SAM 3 Hungarian matcher (`losses/sam3_detection_loss.py`).

The loss terms land in a later step; this module covers the matcher only.

**Every value oracle here is written from the UPSTREAM REFERENCE**
(`sam3/train/matcher.py::BinaryHungarianMatcherV2`, `focal=True, stable=False`,
and `sam3/model/box_ops.py`) at the pinned clone
`96914d2425f90a64f45ca977c2b5165418099543`, transcribed into float64 numpy in
:func:`_reference_cost_matrix` below, and NEVER read off the implementation
under test. The port reaches the reference by exactly one named divergence on
the value path -- padded target columns are costed `INVALID_COST` instead of
being sliced away -- which the oracle applies explicitly rather than inheriting.
"""

import itertools

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from dl_techniques.losses.sam3_detection_loss import (
    INVALID_COST,
    VALID_COST_THRESHOLD,
    _Sam3HungarianMatcher,
    _solve_assignments,
    box_cxcywh_to_xyxy,
    pairwise_generalized_iou,
)

SEED = 20260805


# --------------------------------------------------------------------------
# The reference oracle, transcribed in float64. Nothing below reads the port.
# --------------------------------------------------------------------------

def _ref_cxcywh_to_xyxy(boxes):
    """`box_ops.py::box_cxcywh_to_xyxy`, verbatim."""
    x_c, y_c, w, h = (boxes[..., i] for i in range(4))
    return np.stack([x_c - 0.5 * w, y_c - 0.5 * h,
                     x_c + 0.5 * w, y_c + 0.5 * h], axis=-1)


def _ref_generalized_box_iou(boxes1, boxes2):
    """`box_ops.py::box_iou` + `generalized_box_iou`, verbatim, per image."""
    def area(b):
        return (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])

    area1, area2 = area(boxes1), area(boxes2)
    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union

    lt_e = np.minimum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb_e = np.maximum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh_e = np.clip(rb_e - lt_e, 0.0, None)
    enclose = wh_e[..., 0] * wh_e[..., 1]
    return iou - (enclose - union) / enclose


def _ref_cost_matrix(logits, pred_boxes, tgt_boxes, tgt_valid,
                     cost_class_w=2.0, cost_bbox_w=5.0, cost_giou_w=2.0,
                     alpha=0.25, gamma=2.0):
    """`BinaryHungarianMatcherV2.forward`, the `focal=True, stable=False` path.

    Reference expression, term by term::

        out_prob    = sigmoid(out_score)
        cost_class  = -alpha * (1 - out_prob)**gamma * logsigmoid( out_score)
                      + (1 - alpha) * out_prob**gamma  * logsigmoid(-out_score)
        cost_bbox   = cdist(out_bbox, tgt_bbox, p=1)
        cost_giou   = -generalized_box_iou(xyxy(out), xyxy(tgt))
        C = cost_bbox_w * cost_bbox + cost_class_w * cost_class[..., None]
            + cost_giou_w * cost_giou
        C = where(target_is_valid_padded[:, None, :], C, 1e9)

    `logsigmoid` is spelled here as `-log1p(exp(-x))` for `x >= 0` and
    `x - log1p(exp(x))` for `x < 0` -- the standard branchwise-stable form. It is
    deliberately NOT `-softplus(-x)`, which is the port's spelling; agreeing on a
    spelling would make the comparison circular.
    """
    logits = np.asarray(logits, dtype=np.float64)
    pred_boxes = np.asarray(pred_boxes, dtype=np.float64)
    tgt_boxes = np.asarray(tgt_boxes, dtype=np.float64)
    tgt_valid = np.asarray(tgt_valid, dtype=np.float64)
    if logits.ndim == 3:
        logits = logits[..., 0]

    def logsigmoid(x):
        return np.where(x >= 0.0, -np.log1p(np.exp(-np.abs(x))),
                        x - np.log1p(np.exp(-np.abs(x))))

    prob = 1.0 / (1.0 + np.exp(-logits))
    cost_class = (-alpha * (1.0 - prob) ** gamma * logsigmoid(logits)
                  + (1.0 - alpha) * prob ** gamma * logsigmoid(-logits))

    batch = pred_boxes.shape[0]
    out = np.empty((batch, pred_boxes.shape[1], tgt_boxes.shape[1]),
                   dtype=np.float64)
    for b in range(batch):
        # The reference SLICES padded columns away; this port costs them 1e9.
        # The oracle therefore substitutes a dummy for the sliced columns so the
        # arithmetic is defined, then overwrites them -- the substituted value is
        # never compared.
        boxes_b = np.where(tgt_valid[b][:, None] > 0.0, tgt_boxes[b],
                           np.array([0.5, 0.5, 1.0, 1.0]))
        cost_bbox = np.abs(pred_boxes[b][:, None, :]
                           - boxes_b[None, :, :]).sum(-1)
        cost_giou = -_ref_generalized_box_iou(
            _ref_cxcywh_to_xyxy(pred_boxes[b]), _ref_cxcywh_to_xyxy(boxes_b))
        cost = (cost_bbox_w * cost_bbox
                + cost_class_w * cost_class[b][:, None]
                + cost_giou_w * cost_giou)
        out[b] = np.where(tgt_valid[b][None, :] > 0.0, cost, INVALID_COST)
    return out


# --------------------------------------------------------------------------
# Fixtures / helpers
# --------------------------------------------------------------------------

def _sample(batch, num_queries, num_targets, valid, seed=SEED):
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal((batch, num_queries, 1)).astype("float32")
    pred = (rng.random((batch, num_queries, 4)) * 0.5 + 0.25).astype("float32")
    tgt = (rng.random((batch, num_targets, 4)) * 0.5 + 0.25).astype("float32")
    tgt = tgt * np.asarray(valid, dtype="float32")[:, :, None]
    return logits, pred, tgt, np.asarray(valid, dtype="float32")


# --------------------------------------------------------------------------
# Box geometry
# --------------------------------------------------------------------------

class TestBoxGeometry:

    def test_cxcywh_to_xyxy_matches_reference(self):
        rng = np.random.default_rng(SEED)
        boxes = (rng.random((3, 7, 4)) * 0.8 + 0.1).astype("float64")
        expected = _ref_cxcywh_to_xyxy(boxes)
        actual = ops.convert_to_numpy(
            box_cxcywh_to_xyxy(ops.convert_to_tensor(boxes)))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)

    def test_cxcywh_to_xyxy_on_a_hand_computed_box(self):
        # A value oracle a human can check: centre (0.5, 0.5), 0.2 x 0.4.
        box = np.array([[[0.5, 0.5, 0.2, 0.4]]])
        actual = ops.convert_to_numpy(
            box_cxcywh_to_xyxy(ops.convert_to_tensor(box)))
        np.testing.assert_allclose(actual[0, 0], [0.4, 0.3, 0.6, 0.7],
                                   rtol=0, atol=1e-12)

    def test_giou_of_identical_boxes_is_exactly_one(self):
        boxes = np.array([[[0.0, 0.0, 1.0, 1.0], [0.2, 0.3, 0.9, 0.8]]])
        giou = ops.convert_to_numpy(
            pairwise_generalized_iou(ops.convert_to_tensor(boxes),
                                     ops.convert_to_tensor(boxes)))
        np.testing.assert_allclose(np.diag(giou[0]), [1.0, 1.0],
                                   rtol=0, atol=1e-6)

    def test_giou_of_disjoint_boxes_is_negative_and_hand_computable(self):
        # Two unit-area 1x1 boxes side by side with a 1-unit gap:
        # inter 0, union 2, iou 0, enclosing 3x1 = 3 -> giou = 0 - (3-2)/3.
        a = np.array([[[0.0, 0.0, 1.0, 1.0]]])
        b = np.array([[[2.0, 0.0, 3.0, 1.0]]])
        giou = ops.convert_to_numpy(
            pairwise_generalized_iou(ops.convert_to_tensor(a),
                                     ops.convert_to_tensor(b)))
        np.testing.assert_allclose(giou[0, 0, 0], -1.0 / 3.0,
                                   rtol=0, atol=1e-6)

    def test_giou_matches_reference_oracle(self):
        rng = np.random.default_rng(SEED + 1)
        a = _ref_cxcywh_to_xyxy(
            (rng.random((2, 6, 4)) * 0.5 + 0.25).astype("float64"))
        b = _ref_cxcywh_to_xyxy(
            (rng.random((2, 4, 4)) * 0.5 + 0.25).astype("float64"))
        expected = np.stack([_ref_generalized_box_iou(a[i], b[i])
                             for i in range(2)])
        actual = ops.convert_to_numpy(
            pairwise_generalized_iou(ops.convert_to_tensor(a),
                                     ops.convert_to_tensor(b)))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)

    def test_giou_is_nan_only_when_BOTH_boxes_are_degenerate(self):
        """The measured boundary of `pairwise_generalized_iou`'s domain.

        This pins the fact that killed the matcher's first (removed) dummy-box
        guard, D-006: an all-zero TARGET alone does NOT produce `nan`, because
        `union = area_pred + 0 - 0` is still positive. Only a degenerate box on
        BOTH sides makes union and enclosing area zero at once.

        The second arm is the M3 liveness arm -- without it, a
        `pairwise_generalized_iou` that returned `nan` unconditionally would
        satisfy the first assertion.
        """
        zero = np.zeros((1, 1, 4), dtype="float64")
        real = np.array([[[0.2, 0.2, 0.8, 0.8]]], dtype="float64")

        both_degenerate = ops.convert_to_numpy(pairwise_generalized_iou(
            ops.convert_to_tensor(zero), ops.convert_to_tensor(zero)))
        assert np.isnan(both_degenerate[0, 0, 0])

        one_degenerate = ops.convert_to_numpy(pairwise_generalized_iou(
            ops.convert_to_tensor(real), ops.convert_to_tensor(zero)))
        assert np.isfinite(one_degenerate[0, 0, 0])
        # inter 0, union 0.36, iou 0, enclose 0.8*0.8 = 0.64
        np.testing.assert_allclose(one_degenerate[0, 0, 0],
                                   0.0 - (0.64 - 0.36) / 0.64,
                                   rtol=0, atol=1e-12)


# --------------------------------------------------------------------------
# The cost matrix -- the M1 oracle test
# --------------------------------------------------------------------------

class TestCostMatrix:

    def test_matches_the_reference_oracle(self):
        logits, pred, tgt, valid = _sample(
            3, 6, 4, [[1, 1, 1, 0], [1, 0, 0, 0], [1, 1, 0, 0]])
        matcher = _Sam3HungarianMatcher()
        expected = _ref_cost_matrix(logits, pred, tgt, valid)
        actual = ops.convert_to_numpy(
            matcher.cost_matrix(logits, pred, tgt, valid))
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

    def test_matches_the_reference_oracle_at_non_default_weights(self):
        logits, pred, tgt, valid = _sample(2, 5, 3, [[1, 1, 0], [1, 1, 1]])
        matcher = _Sam3HungarianMatcher(cost_class=1.5, cost_bbox=3.0,
                                        cost_giou=0.5, alpha=0.6, gamma=1.0)
        expected = _ref_cost_matrix(logits, pred, tgt, valid,
                                    cost_class_w=1.5, cost_bbox_w=3.0,
                                    cost_giou_w=0.5, alpha=0.6, gamma=1.0)
        actual = ops.convert_to_numpy(
            matcher.cost_matrix(logits, pred, tgt, valid))
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

    def test_invalid_columns_are_exactly_the_sentinel(self):
        logits, pred, tgt, valid = _sample(2, 4, 5,
                                           [[1, 0, 1, 0, 0], [0, 0, 0, 0, 0]])
        matcher = _Sam3HungarianMatcher()
        cost = ops.convert_to_numpy(
            matcher.cost_matrix(logits, pred, tgt, valid))
        invalid = valid == 0.0
        assert np.all(cost[:, :, invalid[0]][0] == INVALID_COST)
        assert np.all(cost[1] == INVALID_COST)
        assert np.all(cost[0][:, valid[0] > 0] < VALID_COST_THRESHOLD)

    def test_is_finite_even_with_all_zero_padded_target_rows(self):
        logits, pred, tgt, valid = _sample(2, 4, 4, [[1, 0, 0, 0]] * 2)
        cost = ops.convert_to_numpy(
            _Sam3HungarianMatcher().cost_matrix(logits, pred, tgt, valid))
        assert np.all(np.isfinite(cost))

    def test_cost_is_finite_with_a_degenerate_prediction_and_padding(self):
        """The ONLY configuration that can make the raw GIoU `nan` (D-006).

        A zero-width predicted box (reachable only via a float32-underflowed
        sigmoid) paired with an all-zero padded target row gives `0/0`. The
        sentinel `ops.where` overwrites exactly that column, so the returned
        cost is finite -- measured, not assumed. The second arm is the liveness
        check: the VALID column of the same degenerate query must survive as a
        real finite cost, so this cannot pass by everything being sentinel.
        """
        pred = np.array([[[0.5, 0.5, 0.0, 0.0],
                          [0.3, 0.3, 0.2, 0.2]]], dtype="float32")
        tgt = np.array([[[0.3, 0.3, 0.2, 0.2],
                         [0.0, 0.0, 0.0, 0.0]]], dtype="float32")
        valid = np.array([[1.0, 0.0]], dtype="float32")
        logits = np.zeros((1, 2, 1), dtype="float32")
        cost = ops.convert_to_numpy(
            _Sam3HungarianMatcher().cost_matrix(logits, pred, tgt, valid))
        assert np.all(np.isfinite(cost))
        assert np.all(cost[0, :, 1] == INVALID_COST)
        assert np.all(cost[0, :, 0] < VALID_COST_THRESHOLD)

    def test_gradient_through_the_cost_is_finite_with_padded_targets(self):
        logits, pred, tgt, valid = _sample(2, 4, 4, [[1, 1, 0, 0]] * 2)
        pred_t = tf.constant(pred)
        with tf.GradientTape() as tape:
            tape.watch(pred_t)
            cost = _Sam3HungarianMatcher().cost_matrix(
                logits, pred_t, tgt, valid)
            total = ops.sum(ops.where(cost < VALID_COST_THRESHOLD, cost, 0.0))
        grad = tape.gradient(total, pred_t).numpy()
        assert np.all(np.isfinite(grad))

    def test_all_zero_weights_raise(self):
        with pytest.raises(ValueError, match="all three cost weights"):
            _Sam3HungarianMatcher(cost_class=0.0, cost_bbox=0.0,
                                  cost_giou=0.0)

    def test_get_config_round_trips(self):
        matcher = _Sam3HungarianMatcher(cost_class=1.0, cost_bbox=2.0,
                                        cost_giou=3.0, alpha=0.4, gamma=1.5)
        rebuilt = _Sam3HungarianMatcher(**matcher.get_config())
        assert rebuilt.get_config() == matcher.get_config()
        assert matcher.get_config() == {
            "cost_class": 1.0, "cost_bbox": 2.0, "cost_giou": 3.0,
            "alpha": 0.4, "gamma": 1.5}

    def test_shipped_defaults_are_the_reference_values(self):
        # roboflow_v100_full_ft_100_images.yaml:180-188
        assert _Sam3HungarianMatcher().get_config() == {
            "cost_class": 2.0, "cost_bbox": 5.0, "cost_giou": 2.0,
            "alpha": 0.25, "gamma": 2.0}

    def test_accepts_both_rank_2_and_rank_3_logits(self):
        logits, pred, tgt, valid = _sample(2, 4, 3, [[1, 1, 1]] * 2)
        matcher = _Sam3HungarianMatcher()
        a = ops.convert_to_numpy(matcher.cost_matrix(logits, pred, tgt, valid))
        b = ops.convert_to_numpy(
            matcher.cost_matrix(logits[..., 0], pred, tgt, valid))
        np.testing.assert_array_equal(a, b)


# --------------------------------------------------------------------------
# The assignment
# --------------------------------------------------------------------------

class TestAssignment:

    def test_more_queries_than_targets_matches_every_valid_target(self):
        logits, pred, tgt, valid = _sample(2, 8, 3, [[1, 1, 1], [1, 1, 0]])
        _, is_matched = _Sam3HungarianMatcher()(logits, pred, tgt, valid)
        counts = ops.convert_to_numpy(ops.sum(is_matched, axis=1))
        np.testing.assert_array_equal(counts, [3.0, 2.0])

    def test_more_targets_than_queries_matches_every_query(self):
        logits, pred, tgt, valid = _sample(2, 3, 8, [[1] * 8, [1] * 5 + [0] * 3])
        assignment, is_matched = _Sam3HungarianMatcher()(
            logits, pred, tgt, valid)
        counts = ops.convert_to_numpy(ops.sum(is_matched, axis=1))
        np.testing.assert_array_equal(counts, [3.0, 3.0])
        # Every matched target index is distinct within an image.
        idx = ops.convert_to_numpy(assignment)
        for b in range(2):
            chosen = idx[b][ops.convert_to_numpy(is_matched)[b] > 0]
            assert len(set(chosen.tolist())) == len(chosen)

    def test_all_targets_invalid_yields_no_matches_and_does_not_raise(self):
        logits, pred, tgt, valid = _sample(3, 5, 4, [[0, 0, 0, 0]] * 3)
        assignment, is_matched = _Sam3HungarianMatcher()(
            logits, pred, tgt, valid)
        np.testing.assert_array_equal(ops.convert_to_numpy(is_matched),
                                      np.zeros((3, 5), dtype="float32"))
        np.testing.assert_array_equal(ops.convert_to_numpy(assignment),
                                      np.zeros((3, 5), dtype="int32"))

    def test_no_match_ever_lands_on_an_invalid_target(self):
        """The `1e9` cost / `<1e8` filter, exercised where it can actually bite.

        With more queries than valid targets, `linear_sum_assignment` on the
        FULL padded matrix returns `min(Q, N_padded)` pairs, several of which
        necessarily sit on sentinel columns. Only the post-filter removes them.
        """
        valid = [[1, 0, 1, 0, 0], [1, 1, 1, 0, 0]]
        logits, pred, tgt, v = _sample(2, 6, 5, valid)
        assignment, is_matched = _Sam3HungarianMatcher()(logits, pred, tgt, v)
        idx = ops.convert_to_numpy(assignment)
        mask = ops.convert_to_numpy(is_matched)
        for b in range(2):
            for q in range(6):
                if mask[b, q] > 0:
                    assert v[b][idx[b, q]] == 1.0
            assert mask[b].sum() == float(sum(valid[b]))

    def test_assignment_is_globally_optimal_against_brute_force(self):
        logits, pred, tgt, valid = _sample(2, 4, 3, [[1, 1, 1], [1, 1, 1]])
        matcher = _Sam3HungarianMatcher()
        cost = _ref_cost_matrix(logits, pred, tgt, valid)
        assignment, is_matched = matcher(logits, pred, tgt, valid)
        idx = ops.convert_to_numpy(assignment)
        mask = ops.convert_to_numpy(is_matched)
        for b in range(2):
            achieved = sum(cost[b, q, idx[b, q]]
                           for q in range(4) if mask[b, q] > 0)
            best = min(
                sum(cost[b, queries[t], t] for t in range(3))
                for queries in itertools.permutations(range(4), 3))
            assert achieved == pytest.approx(best, rel=1e-9, abs=1e-9)

    def test_permuting_queries_preserves_the_matched_total_cost(self):
        logits, pred, tgt, valid = _sample(1, 6, 4, [[1, 1, 1, 0]])
        matcher = _Sam3HungarianMatcher()
        cost = _ref_cost_matrix(logits, pred, tgt, valid)

        def total(logits_, pred_, cost_):
            assignment, is_matched = matcher(logits_, pred_, tgt, valid)
            idx = ops.convert_to_numpy(assignment)[0]
            mask = ops.convert_to_numpy(is_matched)[0]
            return sum(cost_[0, q, idx[q]]
                       for q in range(len(mask)) if mask[q] > 0)

        base = total(logits, pred, cost)
        perm = np.array([4, 0, 5, 2, 1, 3])
        permuted = total(logits[:, perm], pred[:, perm], cost[:, perm])
        assert permuted == pytest.approx(base, rel=1e-9, abs=1e-9)

    def test_matching_is_exact_when_a_query_reproduces_a_target(self):
        """A hand-constructible assignment: query q must take target q.

        Three queries sit exactly on three targets, in a scrambled order, and
        every other query sits far away with a strongly negative logit.
        """
        targets = np.array([[[0.2, 0.2, 0.1, 0.1],
                             [0.5, 0.5, 0.2, 0.2],
                             [0.8, 0.8, 0.1, 0.1]]], dtype="float32")
        queries = np.array([[targets[0, 2], targets[0, 0],
                             [0.05, 0.95, 0.02, 0.02], targets[0, 1]]],
                           dtype="float32")
        logits = np.zeros((1, 4, 1), dtype="float32")
        valid = np.ones((1, 3), dtype="float32")
        assignment, is_matched = _Sam3HungarianMatcher()(
            logits, queries, targets, valid)
        idx = ops.convert_to_numpy(assignment)[0]
        mask = ops.convert_to_numpy(is_matched)[0]
        np.testing.assert_array_equal(mask, [1.0, 1.0, 0.0, 1.0])
        assert idx[0] == 2 and idx[1] == 0 and idx[3] == 1

    def test_graph_mode_agrees_with_eager_and_keeps_the_static_shape(self):
        logits, pred, tgt, valid = _sample(2, 6, 4, [[1, 1, 0, 0], [1, 1, 1, 1]])
        matcher = _Sam3HungarianMatcher()
        eager = [ops.convert_to_numpy(t)
                 for t in matcher(logits, pred, tgt, valid)]

        @tf.function
        def run(a, b, c, d):
            assignment, is_matched = matcher(a, b, c, d)
            assert tuple(assignment.shape) == (2, 6)
            assert tuple(is_matched.shape) == (2, 6)
            return assignment, is_matched

        graph = [t.numpy() for t in run(tf.constant(logits), tf.constant(pred),
                                        tf.constant(tgt), tf.constant(valid))]
        np.testing.assert_array_equal(graph[0], eager[0])
        np.testing.assert_array_equal(graph[1], eager[1])

    def test_solver_handles_zero_target_columns(self):
        assignment, is_matched = _solve_assignments(
            np.zeros((2, 3, 0), dtype="float64"))
        np.testing.assert_array_equal(is_matched, np.zeros((2, 3)))
        np.testing.assert_array_equal(assignment, np.zeros((2, 3)))


# --------------------------------------------------------------------------
# Gradient reachability through the gathered matched costs
# --------------------------------------------------------------------------

class TestGradient:

    def test_l1_through_the_gathered_match_has_magnitude_exactly_one(self):
        """The consumer-side pattern: gather by the match, then differentiate.

        For `sum |pred - gt|`, `d/d pred` is exactly `+/-1` per coordinate at a
        MATCHED query and exactly `0` at an unmatched one. The zero arm is the
        discriminating one -- a test that only checked "some gradient exists"
        would pass with the match ignored entirely.
        """
        logits, pred, tgt, valid = _sample(2, 5, 3, [[1, 1, 1], [1, 0, 0]])
        matcher = _Sam3HungarianMatcher()
        pred_t = tf.constant(pred)
        with tf.GradientTape() as tape:
            tape.watch(pred_t)
            assignment, is_matched = matcher(logits, pred_t, tgt, valid)
            gathered = tf.gather(tf.constant(tgt), assignment, batch_dims=1)
            l1 = ops.sum(ops.abs(pred_t - gathered), axis=-1)
            loss = ops.sum(l1 * is_matched)
        grad = tape.gradient(loss, pred_t).numpy()
        mask = ops.convert_to_numpy(is_matched)

        assert np.all(np.isfinite(grad))
        matched_rows = grad[mask > 0]
        unmatched_rows = grad[mask == 0]
        np.testing.assert_allclose(np.abs(matched_rows), 1.0,
                                   rtol=0, atol=1e-6)
        np.testing.assert_array_equal(unmatched_rows,
                                      np.zeros_like(unmatched_rows))
        assert matched_rows.size == int(mask.sum()) * 4
        assert matched_rows.size > 0  # liveness: the assertions above are live

    def test_no_gradient_reaches_the_matcher_inputs_through_the_assignment(
            self):
        """The assignment is a hard, non-differentiable decision, by design."""
        logits, pred, tgt, valid = _sample(1, 4, 3, [[1, 1, 1]])
        pred_t = tf.constant(pred)
        with tf.GradientTape() as tape:
            tape.watch(pred_t)
            _, is_matched = _Sam3HungarianMatcher()(logits, pred_t, tgt, valid)
            loss = ops.sum(is_matched)
        assert tape.gradient(loss, pred_t) is None


# --------------------------------------------------------------------------
# M5 -- the same tolerance measured in both precision regimes
# --------------------------------------------------------------------------

class TestPrecisionRegimes:

    @pytest.mark.parametrize("tf32", [True, False])
    def test_cost_matrix_agrees_with_the_oracle_in_both_tf32_regimes(self,
                                                                    tf32):
        """TF32 only rounds matmul inputs; this cost has no matmul at all.

        Recorded rather than assumed: the same tolerance is measured with TF32
        forced ON and forced OFF, because a process-global TF32 toggle elsewhere
        in the suite has swung an unrelated measurement by ~1,775x on this GPU.
        """
        previous = tf.config.experimental.tensor_float_32_execution_enabled()
        tf.config.experimental.enable_tensor_float_32_execution(tf32)
        try:
            logits, pred, tgt, valid = _sample(2, 6, 4,
                                               [[1, 1, 1, 0], [1, 1, 0, 0]])
            expected = _ref_cost_matrix(logits, pred, tgt, valid)
            actual = ops.convert_to_numpy(
                _Sam3HungarianMatcher().cost_matrix(logits, pred, tgt, valid))
            finite = expected < VALID_COST_THRESHOLD
            np.testing.assert_allclose(actual[finite], expected[finite],
                                       rtol=2e-5, atol=2e-5)
        finally:
            tf.config.experimental.enable_tensor_float_32_execution(previous)


# --------------------------------------------------------------------------
# Serialization sanity for the two public helpers
# --------------------------------------------------------------------------

class TestPublicHelperDtypes:

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_box_conversion_preserves_dtype(self, dtype):
        boxes = ops.convert_to_tensor(
            np.ones((1, 2, 4), dtype=dtype))
        assert keras.backend.standardize_dtype(
            box_cxcywh_to_xyxy(boxes).dtype) == dtype
