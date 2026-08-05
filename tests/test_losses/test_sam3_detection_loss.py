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


# ==========================================================================
# THE LOSS
#
# Everything below is the `Sam3DetectionLoss` half of the module. As above,
# every value oracle is a float64 numpy transcription of the UPSTREAM
# REFERENCE at the pinned clone -- `sam3/train/loss/loss_fns.py`
# (`IABCEMdetr.get_loss` :349-507, `Boxes.get_loss` :538-572,
# `Masks.get_loss` :644-717, `sigmoid_focal_loss` :125-175, `_dice_loss`
# :104-122) and `sam3/train/loss/sam3_loss.py` (`_get_num_boxes` :65-81,
# `scale_by_find_batch_size` :192-195) -- and NEVER read off the port.
#
# The oracle deliberately reduces over MATCHED PAIRS ONLY, the way the
# reference's `index_select` does. The port instead keeps the full `(B, Q)`
# tensor and multiplies by the matcher's mask (its declared
# `+ PORT_ONLY(masked-sum)` divergence), so agreement between the two IS the
# proof that the divergence is value-neutral rather than an assumption that
# it is.
# ==========================================================================

from scipy.optimize import linear_sum_assignment as _lsa  # noqa: E402

from dl_techniques.losses.sam3_detection_loss import (  # noqa: E402
    META_IS_EXHAUSTIVE,
    META_KEEP_LOSS,
    META_NUM_BOXES,
    PACKED_BOX_START,
    PACKED_MASK_START,
    PACKED_SCORE_CHANNEL,
    Sam3DetectionLoss,
    binary_cross_entropy_with_logits,
    derive_keep_loss,
    iou_and_generalized_iou,
    packed_channel_count,
    sigmoid_focal_loss,
    unpack_predictions,
    unpack_targets,
)


def _ref_bce(truth, logits):
    """BCE-with-logits, float64, in a spelling the port does NOT use.

    The port spells it `softplus(-x) + x*(1-t)`. This oracle spells it
    `log(1 + exp(x)) - x*t` via `np.logaddexp`, which is algebraically the same
    number by a different route -- agreeing on the spelling would make the
    comparison circular.
    """
    logits = np.asarray(logits, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    return np.logaddexp(0.0, logits) - logits * truth


def _ref_focal(truth, logits, alpha, gamma):
    """`sigmoid_focal_loss.py:19-20`, unreduced, float64."""
    logits = np.asarray(logits, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    prob = 1.0 / (1.0 + np.exp(-logits))
    p_t = prob * truth + (1.0 - prob) * (1.0 - truth)
    alpha_t = alpha * truth + (1.0 - alpha) * (1.0 - truth)
    return alpha_t * ((1.0 - p_t) ** gamma) * _ref_bce(truth, logits)


def _ref_diag_iou_giou(boxes_a_xyxy, boxes_b_xyxy):
    """`box_ops.py::fast_diag_box_iou` / `fast_diag_generalized_box_iou`."""
    a = np.asarray(boxes_a_xyxy, dtype=np.float64)
    b = np.asarray(boxes_b_xyxy, dtype=np.float64)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    lt = np.maximum(a[..., :2], b[..., :2])
    rb = np.minimum(a[..., 2:], b[..., 2:])
    inter = np.clip(rb - lt, 0.0, None).prod(-1)
    lt2 = np.minimum(a[..., :2], b[..., :2])
    rb2 = np.maximum(a[..., 2:], b[..., 2:])
    tot = np.clip(rb2 - lt2, 0.0, None).prod(-1)
    union = area_a + area_b - inter
    iou = inter / union
    return iou, iou - (tot - union) / tot


def _ref_match(logits, pred_boxes, tgt_boxes, tgt_valid):
    """The reference matcher, run in float64, purely as an ORACLE INPUT.

    The assignment is a hard, discrete, non-differentiable decision whose
    correctness is already pinned by the matcher tests above. Recomputing it
    here from `_ref_cost_matrix` (rather than reading it off the port) keeps the
    loss oracles independent of the port end to end.
    """
    cost = _ref_cost_matrix(logits, pred_boxes, tgt_boxes, tgt_valid)
    pairs = []
    for b in range(cost.shape[0]):
        rows, cols = _lsa(cost[b])
        keep = cost[b][rows, cols] < VALID_COST_THRESHOLD
        pairs.append((rows[keep], cols[keep]))
    return pairs


def _ref_terms(packed, weak_loss=False, pad_n_queries=13, pos_weight=10.0,
               alpha=0.25, gamma=2.0, pos_focal=False, presence_alpha=0.5,
               presence_gamma=0.0, focal_alpha=0.25, focal_gamma=2.0,
               include_masks=True):
    """Every reference loss term AND its numerator, in float64.

    Returns a dict with each term plus a ``*_numerator`` for each, so a divisor
    test can divide the SAME numerator by every candidate denominator and show
    the four wrong ones give a different number.
    """
    logits = packed["logits"].astype(np.float64)
    pred_boxes = packed["pred_boxes"].astype(np.float64)
    pred_masks = packed["pred_masks"].astype(np.float64)
    presence = packed["presence"].astype(np.float64)
    tgt_boxes = packed["tgt_boxes"].astype(np.float64)
    tgt_masks = packed["tgt_masks"].astype(np.float64)
    valid = packed["valid"].astype(np.float64)
    keep = packed["keep"].astype(np.float64)
    exhaustive = packed["exhaustive"].astype(np.float64)

    batch, num_queries = logits.shape
    pixels = pred_masks.shape[-1]
    pairs = _ref_match(logits, pred_boxes, tgt_boxes, valid)

    # `_get_num_boxes`, `normalize_by_valid_object_num=True`, clamp min 1.
    num_boxes = max(float((valid > 0).sum()), 1.0)

    target_classes = np.zeros((batch, num_queries), dtype=np.float64)
    positive_target = np.zeros((batch, num_queries), dtype=np.float64)
    l1_sum = 0.0
    giou_sum = 0.0
    mask_sum = 0.0
    dice_sum = 0.0
    prob = 1.0 / (1.0 + np.exp(-logits))
    for b, (rows, cols) in enumerate(pairs):
        if rows.size == 0:
            continue
        target_classes[b, rows] = 1.0
        src = pred_boxes[b, rows]
        tgt = tgt_boxes[b, cols]
        iou, giou = _ref_diag_iou_giou(_ref_cxcywh_to_xyxy(src),
                                       _ref_cxcywh_to_xyxy(tgt))
        soft = np.clip(prob[b, rows] ** alpha * iou ** (1.0 - alpha),
                       0.01, None)
        positive_target[b, rows] = soft
        l1_sum += np.abs(src - tgt).sum()
        giou_sum += (1.0 - giou).sum()
        if include_masks:
            src_m = pred_masks[b, rows]
            tgt_m = tgt_masks[b, cols]
            mask_sum += _ref_focal(tgt_m, src_m, focal_alpha,
                                   focal_gamma).sum()
            p = 1.0 / (1.0 + np.exp(-src_m))
            numer = 2.0 * (p * tgt_m).sum(-1)
            denom = p.sum(-1) + tgt_m.sum(-1)
            dice_sum += (1.0 - (numer + 1.0) / (denom + 1.0)).sum()

    if pos_focal:
        positive_bce = _ref_focal(positive_target, logits, 0.5, gamma)
    else:
        positive_bce = _ref_bce(positive_target, logits)
    loss_bce = positive_bce * target_classes * pos_weight
    loss_bce = loss_bce + (_ref_bce(target_classes, logits)
                           * (1.0 - target_classes) * prob ** gamma)
    loss_bce = loss_bce * keep

    if weak_loss:
        retained = 1.0 - ((exhaustive[:, None] < 0.5).astype(np.float64)
                          * (target_classes < 0.5).astype(np.float64))
        ce_numerator = float((loss_bce * retained).sum())
        ce_divisor = float(retained.sum()) + 1e-6
    else:
        ce_numerator = float(loss_bce.sum())
        if pad_n_queries is None or num_queries >= pad_n_queries:
            ce_divisor = float(batch * num_queries)
        else:
            ce_divisor = float(pad_n_queries * batch)

    presence_numerator = float(
        _ref_focal(keep, presence, presence_alpha, presence_gamma).sum())

    return {
        "num_boxes": num_boxes,
        "target_classes": target_classes,
        "loss_ce": ce_numerator / ce_divisor,
        "loss_ce_numerator": ce_numerator,
        "loss_ce_divisor": ce_divisor,
        "presence_loss": presence_numerator / batch,
        "presence_loss_numerator": presence_numerator,
        "loss_bbox": l1_sum / num_boxes,
        "loss_bbox_numerator": l1_sum,
        "loss_giou": giou_sum / num_boxes,
        "loss_giou_numerator": giou_sum,
        "loss_mask": mask_sum / (num_boxes * pixels),
        "loss_mask_numerator": mask_sum,
        "loss_dice": dice_sum / num_boxes,
        "loss_dice_numerator": dice_sum,
    }


# --------------------------------------------------------------------------
# The fixture. Its numbers are chosen so that the FIVE DIVISORS are pairwise
# distinct and none of them coincides with a round number by accident:
#
#   B = 3, Q = 7, N_max = 5, P = 11, pad_n_queries = 13, num_boxes = 6,
#   is_exhaustive = [1, 0, 1]  ->  retained = 21 - 5 = 16
#
#   #1 num_boxes          =  6
#   #2 num_boxes * P      = 66
#   #3 batch size         =  3
#   #4 pad_n_queries * B  = 39
#   #5 retained + 1e-6    = 16
#
# Five distinct values, so no test below can pass under a wrong divisor.
# --------------------------------------------------------------------------

BATCH, QUERIES, TARGETS, PIXELS, PAD_N_QUERIES = 3, 7, 5, 11, 13
DIVISOR_NUM_BOXES = 6.0
DIVISOR_NUM_BOXES_TIMES_P = 66.0
DIVISOR_BATCH = 3.0
DIVISOR_PAD_TIMES_BATCH = 39.0
DIVISOR_RETAINED = 16.0


def _make_packed(seed=SEED, valid=None, keep=None, exhaustive=None,
                 num_targets=TARGETS, num_queries=QUERIES, batch=BATCH,
                 pixels=PIXELS, logit_scale=1.0):
    """Build the raw (unpacked) fields and their packed tensors together."""
    rng = np.random.default_rng(seed)
    if valid is None:
        valid = np.array([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0], [1, 0, 0, 0, 0]],
                         dtype="float32")
    valid = np.asarray(valid, dtype="float32")
    if exhaustive is None:
        # The divisor fixture's `is_exhaustive = [1, 0, 1]`, which is what
        # makes divisor #5's retained count 16 rather than the full 21. Any
        # other batch size gets an all-exhaustive default.
        exhaustive = (np.array([1.0, 0.0, 1.0], dtype="float32")
                      if batch == BATCH else np.ones(batch, dtype="float32"))
    exhaustive = np.asarray(exhaustive, dtype="float32")

    fields = {
        "logits": (rng.standard_normal((batch, num_queries)) * logit_scale
                   ).astype("float32"),
        "pred_boxes": (rng.random((batch, num_queries, 4)) * 0.5 + 0.25
                       ).astype("float32"),
        "pred_masks": rng.standard_normal(
            (batch, num_queries, pixels)).astype("float32"),
        "presence": rng.standard_normal((batch, 1)).astype("float32"),
        "tgt_boxes": ((rng.random((batch, num_targets, 4)) * 0.5 + 0.25)
                      * valid[:, :, None]).astype("float32"),
        "tgt_masks": ((rng.random((batch, num_targets, pixels)) > 0.5)
                      .astype("float32") * valid[:, :, None]),
        "valid": valid,
        "exhaustive": exhaustive,
    }
    if keep is None:
        keep = np.asarray(ops.convert_to_numpy(
            derive_keep_loss(fields["tgt_boxes"], valid)), dtype="float32")
    fields["keep"] = np.asarray(keep, dtype="float32").reshape(batch, 1)
    return fields


def _pack(fields, include_masks=True):
    """Assemble the module's packed `(y_true, y_pred)` from raw fields."""
    batch, num_queries = fields["logits"].shape
    num_targets = fields["valid"].shape[1]
    pixels = fields["pred_masks"].shape[-1] if include_masks else 0
    channels = packed_channel_count(pixels)

    y_pred = np.zeros((batch, num_queries + 1, channels), dtype="float32")
    y_pred[:, :num_queries, PACKED_SCORE_CHANNEL] = fields["logits"]
    y_pred[:, :num_queries, PACKED_BOX_START:PACKED_MASK_START] = \
        fields["pred_boxes"]
    if include_masks:
        y_pred[:, :num_queries, PACKED_MASK_START:] = fields["pred_masks"]
    y_pred[:, num_queries, PACKED_SCORE_CHANNEL] = fields["presence"][:, 0]

    y_true = np.zeros((batch, num_targets + 1, channels), dtype="float32")
    y_true[:, :num_targets, PACKED_SCORE_CHANNEL] = fields["valid"]
    y_true[:, :num_targets, PACKED_BOX_START:PACKED_MASK_START] = \
        fields["tgt_boxes"]
    if include_masks:
        y_true[:, :num_targets, PACKED_MASK_START:] = fields["tgt_masks"]
    y_true[:, num_targets, META_KEEP_LOSS] = fields["keep"][:, 0]
    y_true[:, num_targets, META_NUM_BOXES] = fields["valid"].sum(1)
    y_true[:, num_targets, META_IS_EXHAUSTIVE] = fields["exhaustive"]
    return y_true, y_pred


def _loss(**kwargs):
    defaults = dict(include_masks=True, pad_n_queries=PAD_N_QUERIES)
    defaults.update(kwargs)
    return Sam3DetectionLoss(**defaults)


def _terms(loss, y_true, y_pred):
    return {k: float(ops.convert_to_numpy(v))
            for k, v in loss.compute_terms(y_true, y_pred).items()}


# --------------------------------------------------------------------------
# The packed layout
# --------------------------------------------------------------------------

class TestPackedLayout:

    def test_channel_count_is_five_plus_the_mask_size(self):
        assert packed_channel_count(0) == 5
        assert packed_channel_count(11) == 16

    @pytest.mark.parametrize("include_masks", [True, False])
    def test_unpack_predictions_round_trips_value_exactly(self, include_masks):
        fields = _make_packed()
        _, y_pred = _pack(fields, include_masks=include_masks)
        out = unpack_predictions(ops.convert_to_tensor(y_pred), include_masks)
        np.testing.assert_array_equal(
            ops.convert_to_numpy(out["pred_logits"]), fields["logits"])
        np.testing.assert_array_equal(
            ops.convert_to_numpy(out["pred_boxes"]), fields["pred_boxes"])
        np.testing.assert_array_equal(
            ops.convert_to_numpy(out["presence_logit"]), fields["presence"])
        if include_masks:
            np.testing.assert_array_equal(
                ops.convert_to_numpy(out["pred_masks"]), fields["pred_masks"])
        else:
            assert out["pred_masks"] is None

    @pytest.mark.parametrize("include_masks", [True, False])
    def test_unpack_targets_round_trips_value_exactly(self, include_masks):
        fields = _make_packed()
        y_true, _ = _pack(fields, include_masks=include_masks)
        out = unpack_targets(ops.convert_to_tensor(y_true), include_masks)
        np.testing.assert_array_equal(
            ops.convert_to_numpy(out["target_valid"]), fields["valid"])
        np.testing.assert_array_equal(
            ops.convert_to_numpy(out["target_boxes"]), fields["tgt_boxes"])
        np.testing.assert_array_equal(
            ops.convert_to_numpy(out["keep_loss"]), fields["keep"])
        np.testing.assert_array_equal(
            ops.convert_to_numpy(out["num_boxes"]), fields["valid"].sum(1))
        np.testing.assert_array_equal(
            ops.convert_to_numpy(out["is_exhaustive"]), fields["exhaustive"])
        if include_masks:
            np.testing.assert_array_equal(
                ops.convert_to_numpy(out["target_masks"]), fields["tgt_masks"])
        else:
            assert out["target_masks"] is None

    def test_the_presence_row_is_the_last_row_not_the_first(self):
        """A layout test that a transposed convention would fail.

        Reading row 0 instead of row Q would silently pick up a query's logit,
        which is a real number and would look plausible forever.
        """
        fields = _make_packed()
        _, y_pred = _pack(fields)
        out = unpack_predictions(ops.convert_to_tensor(y_pred), True)
        assert not np.allclose(fields["presence"][:, 0], fields["logits"][:, 0])
        np.testing.assert_array_equal(
            ops.convert_to_numpy(out["presence_logit"])[:, 0],
            fields["presence"][:, 0])


class TestDeriveKeepLoss:

    def test_an_image_with_no_valid_target_is_a_presence_negative(self):
        boxes = np.zeros((1, 3, 4), dtype="float32")
        valid = np.zeros((1, 3), dtype="float32")
        assert float(ops.convert_to_numpy(derive_keep_loss(boxes, valid))) == 0.0

    def test_an_image_with_one_valid_target_is_a_presence_positive(self):
        """The M3 liveness arm for the test above: not always-zero."""
        boxes = np.array([[[0.5, 0.5, 0.2, 0.2], [0, 0, 0, 0]]],
                         dtype="float32")
        valid = np.array([[1.0, 0.0]], dtype="float32")
        assert float(ops.convert_to_numpy(derive_keep_loss(boxes, valid))) == 1.0

    def test_a_valid_row_with_a_zero_area_box_does_NOT_count_as_visible(self):
        """The `w > 0 & h > 0` terms of `loss_fns.py:418-422` are LIVE.

        An "invisible object" row -- a real instance whose box has collapsed --
        must read as ABSENT. Dropping those two terms would flip this image to a
        presence positive with nothing to detect.
        """
        boxes = np.array([[[0.5, 0.5, 0.0, 0.3]]], dtype="float32")
        valid = np.ones((1, 1), dtype="float32")
        assert float(ops.convert_to_numpy(derive_keep_loss(boxes, valid))) == 0.0
        boxes_h = np.array([[[0.5, 0.5, 0.3, 0.0]]], dtype="float32")
        assert float(
            ops.convert_to_numpy(derive_keep_loss(boxes_h, valid))) == 0.0


# --------------------------------------------------------------------------
# SC-6 -- the BCE spelling and the gradient at EXACTLY x = 0
# --------------------------------------------------------------------------

class TestBceSpelling:

    @pytest.mark.parametrize("logit", [-1024.0, -70.0, -8.0, -1e-8, 0.0, 1e-8,
                                       8.0, 70.0, 1024.0])
    @pytest.mark.parametrize("truth", [0.0, 1.0, 0.37])
    def test_bce_value_matches_an_independently_spelled_oracle(self, logit,
                                                               truth):
        actual = float(ops.convert_to_numpy(binary_cross_entropy_with_logits(
            ops.convert_to_tensor(np.float64(truth)),
            ops.convert_to_tensor(np.float64(logit)))))
        np.testing.assert_allclose(actual, _ref_bce(truth, logit),
                                   rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("truth,expected", [(1.0, -0.5), (0.0, 0.5)])
    def test_gradient_at_exactly_zero_logit_is_plus_or_minus_one_half(
            self, truth, expected):
        """SC-6. The whole reason the `softplus` spelling is mandatory.

        `d/dx BCE(t, x) = sigmoid(x) - t`, which at `x = 0` is exactly
        `0.5 - t`: `-0.5` at `t = 1` and `+0.5` at `t = 0`. An exactly-zero
        logit is not hypothetical -- this module multiplies BCE by `ops.where`
        gates that emit exact zeros, and a zero-initialized head emits exact
        zeros at step 0.
        """
        x = tf.constant(0.0, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x)
            value = binary_cross_entropy_with_logits(
                ops.convert_to_tensor(truth), x)
        grad = float(ops.convert_to_numpy(tape.gradient(value, x)))
        assert grad == pytest.approx(expected, abs=1e-7)

    @pytest.mark.parametrize("truth,textbook,correct",
                             [(1.0, 0.0, -0.5), (0.0, 1.0, 0.5)])
    def test_the_textbook_spelling_really_is_wrong_at_zero(self, truth,
                                                           textbook, correct):
        """The M3 liveness arm: the assertion above DISCRIMINATES.

        Without this, `test_gradient_at_exactly_zero_logit...` would be
        satisfied by any implementation that happens to be right, with no
        evidence that a wrong one exists to be caught. Here the rejected
        spelling is executed and measured: `max(x, 0) - x*t + log1p(exp(-|x|))`
        autodiffs to `+1.0` at `t = 0` and `0.0` at `t = 1` -- a SIGN FLIP on
        the positive branch, not a rounding difference.
        """
        x = tf.constant(0.0, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x)
            value = (ops.maximum(x, 0.0) - x * truth
                     + ops.log1p(ops.exp(-ops.abs(x))))
        grad = float(ops.convert_to_numpy(tape.gradient(value, x)))
        assert grad == pytest.approx(textbook, abs=1e-7)
        assert abs(grad - correct) > 0.4


class TestFocalSpelling:

    def test_gamma_zero_is_exactly_plain_alpha_weighted_bce(self):
        """S-2: the shipped `presence_gamma` is 0.0, so the focal modulation
        is INERT and the presence term IS plain alpha-weighted BCE. That is
        reproduced faithfully rather than "improved"."""
        rng = np.random.default_rng(SEED)
        logits = rng.standard_normal(64).astype("float64")
        truth = (rng.random(64) > 0.5).astype("float64")
        alpha = 0.5
        actual = ops.convert_to_numpy(sigmoid_focal_loss(
            ops.convert_to_tensor(truth), ops.convert_to_tensor(logits),
            alpha=alpha, gamma=0.0))
        alpha_t = alpha * truth + (1.0 - alpha) * (1.0 - truth)
        np.testing.assert_allclose(actual, alpha_t * _ref_bce(truth, logits),
                                   rtol=1e-12, atol=1e-12)

    def test_gamma_two_is_NOT_plain_bce(self):
        """The M3 liveness arm: the modulator is inert only AT gamma = 0."""
        rng = np.random.default_rng(SEED)
        logits = rng.standard_normal(64).astype("float64")
        truth = (rng.random(64) > 0.5).astype("float64")
        alpha = 0.5
        actual = ops.convert_to_numpy(sigmoid_focal_loss(
            ops.convert_to_tensor(truth), ops.convert_to_tensor(logits),
            alpha=alpha, gamma=2.0))
        alpha_t = alpha * truth + (1.0 - alpha) * (1.0 - truth)
        assert not np.allclose(actual, alpha_t * _ref_bce(truth, logits))
        np.testing.assert_allclose(actual, _ref_focal(truth, logits, alpha,
                                                      2.0),
                                   rtol=1e-12, atol=1e-12)

    def test_focal_uses_alpha_t_not_a_scalar_alpha(self):
        """SAM 2's `_focal_from_logits` applies alpha as a FLAT SCALAR (its
        D-073). SAM 3's reference does not. A port that copied SAM 2's
        arithmetic wholesale would down-weight one class by 3x."""
        logits = np.array([0.7], dtype="float64")
        alpha = 0.25
        positive = float(ops.convert_to_numpy(sigmoid_focal_loss(
            ops.convert_to_tensor(np.array([1.0])),
            ops.convert_to_tensor(logits), alpha=alpha, gamma=2.0)))
        negative = float(ops.convert_to_numpy(sigmoid_focal_loss(
            ops.convert_to_tensor(np.array([0.0])),
            ops.convert_to_tensor(logits), alpha=alpha, gamma=2.0)))
        # Under a flat scalar alpha both would carry the same 0.25 factor.
        np.testing.assert_allclose(
            positive, alpha * (1 - 1 / (1 + np.exp(-logits[0]))) ** 2
            * _ref_bce(1.0, logits[0]), rtol=1e-12)
        np.testing.assert_allclose(
            negative, (1 - alpha) * (1 / (1 + np.exp(-logits[0]))) ** 2
            * _ref_bce(0.0, logits[0]), rtol=1e-12)

    def test_gamma_zero_gradient_is_finite_at_a_saturated_logit(self):
        """Why `gamma == 0.0` is special-cased instead of `power(x, 0.0)`.

        At `p_t -> 1` the modulator base is 0, and `d/dp power(1-p_t, 0.0)` is
        `0 * (1-p_t)**-1`, i.e. `0 * inf = nan`. The VALUE path is identical
        either way -- this is a gradient-only defect, which is exactly the kind
        that survives a value test.
        """
        x = tf.constant([40.0], dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x)
            value = ops.sum(sigmoid_focal_loss(
                ops.convert_to_tensor(np.array([1.0], dtype="float32")), x,
                alpha=0.5, gamma=0.0))
        grad = ops.convert_to_numpy(tape.gradient(value, x))
        assert np.all(np.isfinite(grad))


class TestDiagonalGeometry:

    def test_diag_iou_and_giou_match_the_reference_oracle(self):
        rng = np.random.default_rng(SEED + 7)
        a = _ref_cxcywh_to_xyxy((rng.random((3, 9, 4)) * 0.5 + 0.25))
        b = _ref_cxcywh_to_xyxy((rng.random((3, 9, 4)) * 0.5 + 0.25))
        exp_iou, exp_giou = _ref_diag_iou_giou(a, b)
        iou, giou = iou_and_generalized_iou(ops.convert_to_tensor(a),
                                            ops.convert_to_tensor(b))
        np.testing.assert_allclose(ops.convert_to_numpy(iou), exp_iou,
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(ops.convert_to_numpy(giou), exp_giou,
                                   rtol=0, atol=1e-12)

    def test_the_pairwise_adapter_still_agrees_with_the_matcher_oracle(self):
        """The shared-core refactor is value-neutral for the matcher path."""
        rng = np.random.default_rng(SEED + 8)
        a = _ref_cxcywh_to_xyxy((rng.random((2, 6, 4)) * 0.5 + 0.25))
        b = _ref_cxcywh_to_xyxy((rng.random((2, 4, 4)) * 0.5 + 0.25))
        expected = np.stack([_ref_generalized_box_iou(a[i], b[i])
                             for i in range(2)])
        actual = ops.convert_to_numpy(
            pairwise_generalized_iou(ops.convert_to_tensor(a),
                                     ops.convert_to_tensor(b)))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


# --------------------------------------------------------------------------
# The six terms, against the reference oracle
# --------------------------------------------------------------------------

class TestTermValues:

    def test_every_term_matches_the_reference_oracle(self):
        fields = _make_packed()
        y_true, y_pred = _pack(fields)
        actual = _terms(_loss(), y_true, y_pred)
        expected = _ref_terms(fields)
        for key in ("loss_ce", "presence_loss", "loss_bbox", "loss_giou",
                    "loss_mask", "loss_dice"):
            np.testing.assert_allclose(actual[key], expected[key],
                                       rtol=2e-5, atol=2e-6,
                                       err_msg=f"term {key}")
        assert actual["num_boxes"] == DIVISOR_NUM_BOXES
        assert actual["num_matched"] == DIVISOR_NUM_BOXES

    def test_the_positive_branch_is_plain_bce_not_focal_at_pos_focal_false(
            self):
        """S-2. The shipped config sets `pos_focal: false`, so the positive
        classification term is PLAIN BCE times `pos_weight`. Silently applying
        the full focal modulation there -- because the code path is shared --
        is the exact divergence S-2 warns about."""
        fields = _make_packed()
        y_true, y_pred = _pack(fields)
        plain = _terms(_loss(pos_focal=False), y_true, y_pred)["loss_ce"]
        focal = _terms(_loss(pos_focal=True), y_true, y_pred)["loss_ce"]
        np.testing.assert_allclose(
            plain, _ref_terms(fields, pos_focal=False)["loss_ce"],
            rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(
            focal, _ref_terms(fields, pos_focal=True)["loss_ce"],
            rtol=2e-5, atol=2e-6)
        assert abs(plain - focal) > 1e-3  # the switch is LIVE, not cosmetic

    def test_the_iou_aware_soft_target_uses_one_minus_alpha_on_the_iou(self):
        """`t = prob**alpha * iou**(1 - alpha)`, `loss_fns.py:371`.

        Swapping the two exponents is a one-character edit that leaves every
        shape, every sign and every finiteness property intact. It is caught
        here by comparing against an oracle that spells the exponents out.
        """
        fields = _make_packed()
        y_true, y_pred = _pack(fields)
        actual = _terms(_loss(), y_true, y_pred)["loss_ce"]
        correct = _ref_terms(fields, alpha=0.25)["loss_ce"]
        np.testing.assert_allclose(actual, correct, rtol=2e-5, atol=2e-6)
        # An oracle with the exponents swapped: alpha 0.75 puts 0.25 on the IoU.
        swapped = _ref_terms(fields, alpha=0.75)["loss_ce"]
        assert abs(correct - swapped) > 1e-4

    def test_the_soft_target_is_clamped_at_a_floor_of_one_hundredth(self):
        """`torch.clamp(t, 0.01)`, `loss_fns.py:372`. With very negative
        logits `prob**0.25` collapses and the clamp is what stops the positive
        target -- and with it the positive gradient -- from vanishing."""
        fields = _make_packed(logit_scale=1.0)
        fields["logits"] = np.full_like(fields["logits"], -30.0)
        y_true, y_pred = _pack(fields)
        actual = _terms(_loss(), y_true, y_pred)["loss_ce"]
        np.testing.assert_allclose(actual, _ref_terms(fields)["loss_ce"],
                                   rtol=2e-5, atol=2e-6)
        # Liveness: without a clamp the target would be ~0 and the positive
        # term would be ~0 * pos_weight, a materially smaller number.
        assert actual > 0.0

    def test_the_soft_target_is_detached_so_boxes_get_no_gradient_from_ce(
            self):
        """`loss_fns.py:355-374` computes the whole soft target inside
        `torch.no_grad()`. `pred_boxes` therefore reaches `loss_ce` ONLY
        through a stopped gradient, so a CE-only loss must leave the box
        channels with exactly zero gradient."""
        fields = _make_packed()
        y_true, y_pred = _pack(fields)
        loss = _loss(weight_ce=1.0, weight_presence=0.0, weight_bbox=0.0,
                     weight_giou=0.0, weight_mask=0.0, weight_dice=0.0)
        tensor = tf.constant(y_pred)
        with tf.GradientTape() as tape:
            tape.watch(tensor)
            value = loss(y_true, tensor)
        grad = ops.convert_to_numpy(tape.gradient(value, tensor))
        box_grad = grad[:, :QUERIES, PACKED_BOX_START:PACKED_MASK_START]
        np.testing.assert_array_equal(box_grad, np.zeros_like(box_grad))
        # Liveness arm: the SAME probe sees a live gradient on the logits.
        assert np.abs(grad[:, :QUERIES, PACKED_SCORE_CHANNEL]).max() > 1e-6


# --------------------------------------------------------------------------
# SC-4 -- THE FIVE DIVISORS, each as a VALUE ORACLE
#
# Every test below divides the term's OWN numerator by all five candidate
# denominators and asserts the port lands on exactly one of them. The fixture
# makes the five pairwise distinct (6, 66, 3, 39, 16), so a test that passed
# under two candidates -- i.e. that measured nothing -- is impossible here.
# --------------------------------------------------------------------------

class TestTheFiveDivisors:

    def _assert_only(self, actual, numerator, correct, wrong):
        np.testing.assert_allclose(actual, numerator / correct,
                                   rtol=2e-5, atol=2e-7)
        for candidate in wrong:
            assert abs(actual - numerator / candidate) > 1e-4, (
                f"divisor {candidate} is NOT separated from {correct}")

    def test_divisor_1_boxes_and_dice_divide_by_num_boxes(self):
        fields = _make_packed()
        y_true, y_pred = _pack(fields)
        actual = _terms(_loss(), y_true, y_pred)
        expected = _ref_terms(fields)
        wrong = [DIVISOR_NUM_BOXES_TIMES_P, DIVISOR_BATCH,
                 DIVISOR_PAD_TIMES_BATCH, DIVISOR_RETAINED]
        for key in ("loss_bbox", "loss_giou", "loss_dice"):
            self._assert_only(actual[key], expected[f"{key}_numerator"],
                              DIVISOR_NUM_BOXES, wrong)

    def test_divisor_2_the_mask_focal_divides_by_num_boxes_TIMES_pixels(self):
        """The fourth divisor the prior plan never flagged.

        `loss_fns.py:153-155`: the default triton path returns an
        already-summed scalar and divides by `num_boxes * inputs.shape[1]`.
        Dividing by `num_boxes` alone inflates this term by a factor of P --
        11 here, 4096 at a 64x64 mask -- BEFORE its 200.0 weight.
        """
        fields = _make_packed()
        y_true, y_pred = _pack(fields)
        actual = _terms(_loss(), y_true, y_pred)["loss_mask"]
        numerator = _ref_terms(fields)["loss_mask_numerator"]
        self._assert_only(actual, numerator, DIVISOR_NUM_BOXES_TIMES_P,
                          [DIVISOR_NUM_BOXES, DIVISOR_BATCH,
                           DIVISOR_PAD_TIMES_BATCH, DIVISOR_RETAINED])

    def test_divisor_2_and_1_put_the_two_mask_terms_on_different_scales(self):
        """The asymmetry is the REFERENCE's, not a slip: dice reduces
        spatially first, so it carries no extra P factor."""
        fields = _make_packed()
        y_true, y_pred = _pack(fields)
        actual = _terms(_loss(), y_true, y_pred)
        expected = _ref_terms(fields)
        ratio = ((expected["loss_mask_numerator"] / actual["loss_mask"])
                 / (expected["loss_dice_numerator"] / actual["loss_dice"]))
        np.testing.assert_allclose(ratio, PIXELS, rtol=1e-4)

    def test_divisor_3_presence_divides_by_the_BATCH_SIZE(self):
        """`loss_fns.py:433-440` passes `num_boxes=bs` with the explicit
        comment "not num_boxes, but we'll use it to normalize by bs"."""
        fields = _make_packed()
        y_true, y_pred = _pack(fields)
        actual = _terms(_loss(), y_true, y_pred)["presence_loss"]
        numerator = _ref_terms(fields)["presence_loss_numerator"]
        self._assert_only(actual, numerator, DIVISOR_BATCH,
                          [DIVISOR_NUM_BOXES, DIVISOR_NUM_BOXES_TIMES_P,
                           DIVISOR_PAD_TIMES_BATCH, DIVISOR_RETAINED])

    def test_divisor_4_the_SHIPPED_ce_path_divides_by_pad_n_queries_times_B(
            self):
        """`weak_loss: False`, `pad_n_queries: 200` -- the released recipe.
        `loss_fns.py:507`."""
        fields = _make_packed()
        y_true, y_pred = _pack(fields)
        actual = _terms(_loss(weak_loss=False), y_true, y_pred)["loss_ce"]
        reference = _ref_terms(fields, weak_loss=False)
        assert reference["loss_ce_divisor"] == DIVISOR_PAD_TIMES_BATCH
        self._assert_only(actual, reference["loss_ce_numerator"],
                          DIVISOR_PAD_TIMES_BATCH,
                          [DIVISOR_NUM_BOXES, DIVISOR_NUM_BOXES_TIMES_P,
                           DIVISOR_BATCH, DIVISOR_RETAINED])

    def test_divisor_4_falls_back_to_a_plain_mean_when_Q_reaches_the_budget(
            self):
        """`loss_fns.py:500-501`: `pad_n_queries=None` OR `Q >= pad_n_queries`
        is a plain mean over `B * Q` -- here 21, distinct from all five."""
        fields = _make_packed()
        y_true, y_pred = _pack(fields)
        numerator = _ref_terms(fields, pad_n_queries=None)["loss_ce_numerator"]
        for pad in (None, QUERIES, QUERIES - 1):
            actual = _terms(_loss(pad_n_queries=pad), y_true,
                            y_pred)["loss_ce"]
            self._assert_only(actual, numerator, float(BATCH * QUERIES),
                              [DIVISOR_NUM_BOXES, DIVISOR_NUM_BOXES_TIMES_P,
                               DIVISOR_BATCH, DIVISOR_PAD_TIMES_BATCH,
                               DIVISOR_RETAINED])

    def test_divisor_5_the_weak_ce_path_divides_by_the_RETAINED_count(self):
        """`weak_loss=True` is the reference CLASS default (S-1) and it is a
        DIFFERENT denominator, not a stylistic variant: a non-exhaustively
        annotated image's negative supervision leaves the numerator AND the
        denominator, which is what makes this a mean over retained elements
        rather than a down-weighted mean over all of them."""
        fields = _make_packed()
        y_true, y_pred = _pack(fields)
        actual = _terms(_loss(weak_loss=True), y_true, y_pred)["loss_ce"]
        reference = _ref_terms(fields, weak_loss=True)
        # is_exhaustive = [1, 0, 1] and image 1 has 2 matched queries, so its
        # 5 negatives are dropped: 3*7 - 5 = 16.
        np.testing.assert_allclose(reference["loss_ce_divisor"],
                                   DIVISOR_RETAINED + 1e-6, rtol=0, atol=1e-9)
        self._assert_only(actual, reference["loss_ce_numerator"],
                          DIVISOR_RETAINED,
                          [DIVISOR_NUM_BOXES, DIVISOR_NUM_BOXES_TIMES_P,
                           DIVISOR_BATCH, DIVISOR_PAD_TIMES_BATCH])

    def test_the_weak_and_shipped_ce_paths_disagree_on_the_same_inputs(self):
        """S-1's whole point: this is a REAL switch with two documented
        denominators, so neither branch is dead code."""
        fields = _make_packed()
        y_true, y_pred = _pack(fields)
        weak = _terms(_loss(weak_loss=True), y_true, y_pred)["loss_ce"]
        shipped = _terms(_loss(weak_loss=False), y_true, y_pred)["loss_ce"]
        assert abs(weak - shipped) > 1e-3

    def test_is_exhaustive_actually_moves_the_weak_path(self):
        """The M3 liveness arm for divisor 5: the meta channel is READ.

        With every image exhaustive nothing is dropped, the retained count is
        the full `B * Q = 21`, and the term changes. A weak path that ignored
        `is_exhaustive` would return the same number for both.
        """
        mixed = _make_packed()
        all_exhaustive = _make_packed(exhaustive=np.ones(BATCH, "float32"))
        a = _terms(_loss(weak_loss=True), *_pack(mixed))
        b = _terms(_loss(weak_loss=True), *_pack(all_exhaustive))
        assert abs(a["loss_ce"] - b["loss_ce"]) > 1e-4
        np.testing.assert_allclose(
            b["loss_ce"],
            _ref_terms(all_exhaustive, weak_loss=True)["loss_ce"],
            rtol=2e-5, atol=2e-6)

    def test_the_num_boxes_clamp_makes_an_ALL_NEGATIVE_batch_finite(self):
        """H-6. Divisor #1 is `max(sum_valid, 1)`; without the clamp an
        all-negative batch divides by zero and the step NaNs."""
        zeros = np.zeros((BATCH, TARGETS), dtype="float32")
        fields = _make_packed(valid=zeros)
        y_true, y_pred = _pack(fields)
        actual = _terms(_loss(), y_true, y_pred)
        assert actual["num_boxes"] == 1.0
        assert actual["num_matched"] == 0.0
        for key, value in actual.items():
            assert np.isfinite(value), f"{key} is not finite"

    def test_num_boxes_can_also_be_read_from_the_meta_row(self):
        """`normalize_by_valid_object_num=False` is the reference's other
        mode (`sam3_loss.py:72-73`) and gives the meta channel a consumer."""
        fields = _make_packed()
        y_true, y_pred = _pack(fields)
        from_flags = _terms(_loss(normalize_by_valid_object_num=True),
                            y_true, y_pred)["num_boxes"]
        from_meta = _terms(_loss(normalize_by_valid_object_num=False),
                           y_true, y_pred)["num_boxes"]
        assert from_flags == from_meta == DIVISOR_NUM_BOXES
        # Liveness: the two paths read DIFFERENT channels, so a meta row that
        # disagrees with the flags is visible.
        y_true_bad = np.array(y_true)
        y_true_bad[:, TARGETS, META_NUM_BOXES] = 2.0
        assert _terms(_loss(normalize_by_valid_object_num=False),
                      y_true_bad, y_pred)["num_boxes"] == 6.0
        y_true_bad[:, TARGETS, META_NUM_BOXES] = 1.0
        assert _terms(_loss(normalize_by_valid_object_num=False),
                      y_true_bad, y_pred)["num_boxes"] == 3.0


# --------------------------------------------------------------------------
# SC-5 -- presence gates classification MULTIPLICATIVELY, before reduction
# --------------------------------------------------------------------------

class TestPresenceGate:

    def test_presence_gate_zeroes_classification_for_a_zero_gt_image(self):
        """H-7, `loss_fns.py:425`. RED-PROVEN: deleting `* keep_loss` from
        `compute_terms` makes this fail (mutation M-1, burned).

        Construction: image 1 is a zero-GT image, so its `keep_loss` is 0 and
        its ENTIRE classification contribution -- positive and negative -- must
        vanish. The probe replaces that image's logits with a wildly different
        draw: under the gate `loss_ce` cannot move at all, and without the gate
        it moves by a large margin.
        """
        valid = np.array([[1, 1, 1, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 0]],
                         dtype="float32")
        fields = _make_packed(valid=valid)
        assert float(fields["keep"][1, 0]) == 0.0
        y_true, y_pred = _pack(fields)
        base = _terms(_loss(), y_true, y_pred)["loss_ce"]

        perturbed = np.array(y_pred)
        perturbed[1, :QUERIES, PACKED_SCORE_CHANNEL] += 7.5
        moved = _terms(_loss(), y_true, perturbed)["loss_ce"]
        assert moved == pytest.approx(base, rel=1e-6, abs=1e-7)

        # M3 liveness arm: the SAME perturbation on a KEPT image does move it,
        # so the assertion above is measuring the gate and not a dead probe.
        perturbed_kept = np.array(y_pred)
        perturbed_kept[0, :QUERIES, PACKED_SCORE_CHANNEL] += 7.5
        assert abs(_terms(_loss(), y_true, perturbed_kept)["loss_ce"]
                   - base) > 1e-2

    def test_the_gate_is_applied_BEFORE_the_reduction_not_after(self):
        """The gated image's rows still count in divisor #4's denominator.

        A gate applied to the total instead of to the per-element tensor would
        scale by `mean(keep_loss)` and silently rescale every OTHER image's
        contribution. The oracle reduces over `pad_n_queries * B` including the
        gated rows, so agreement pins the ORDER, not merely the presence, of
        the multiply.
        """
        valid = np.array([[1, 1, 1, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 0]],
                         dtype="float32")
        fields = _make_packed(valid=valid)
        y_true, y_pred = _pack(fields)
        actual = _terms(_loss(), y_true, y_pred)["loss_ce"]
        expected = _ref_terms(fields)
        assert expected["loss_ce_divisor"] == DIVISOR_PAD_TIMES_BATCH
        np.testing.assert_allclose(actual, expected["loss_ce"],
                                   rtol=2e-5, atol=2e-6)
        # A post-reduction gate would instead give numerator/(39) * mean(keep).
        post_hoc = expected["loss_ce"] * float(fields["keep"].mean())
        assert abs(actual - post_hoc) > 1e-3

    def test_presence_still_supervises_the_zero_gt_image(self):
        """The gated image is not unsupervised -- `presence_loss` is what
        teaches it to say "absent"."""
        valid = np.array([[1, 1, 1, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 0]],
                         dtype="float32")
        fields = _make_packed(valid=valid)
        y_true, y_pred = _pack(fields)
        base = _terms(_loss(), y_true, y_pred)["presence_loss"]
        perturbed = np.array(y_pred)
        perturbed[1, QUERIES, PACKED_SCORE_CHANNEL] += 5.0
        assert abs(_terms(_loss(), y_true, perturbed)["presence_loss"]
                   - base) > 1e-3

    def test_box_and_mask_terms_contribute_nothing_for_a_zero_gt_image(self):
        valid = np.array([[1, 1, 1, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 0]],
                         dtype="float32")
        fields = _make_packed(valid=valid)
        y_true, y_pred = _pack(fields)
        base = _terms(_loss(), y_true, y_pred)
        perturbed = np.array(y_pred)
        perturbed[1, :QUERIES, PACKED_BOX_START:] += 0.1
        moved = _terms(_loss(), y_true, perturbed)
        for key in ("loss_bbox", "loss_giou", "loss_mask", "loss_dice"):
            assert moved[key] == pytest.approx(base[key], rel=1e-5, abs=1e-7)
        # Liveness: the same perturbation on image 0 moves all four.
        perturbed_kept = np.array(y_pred)
        perturbed_kept[0, :QUERIES, PACKED_BOX_START:] += 0.1
        moved_kept = _terms(_loss(), y_true, perturbed_kept)
        for key in ("loss_bbox", "loss_giou", "loss_mask", "loss_dice"):
            assert abs(moved_kept[key] - base[key]) > 1e-4, key


# --------------------------------------------------------------------------
# First-class cases (not "edge" cases)
# --------------------------------------------------------------------------

class TestFirstClassCases:

    def test_more_gt_than_queries_leaves_the_surplus_unmatched(self):
        """`N > Q`: `linear_sum_assignment` on a rectangular cost assigns
        `min(Q, N_valid)` pairs; the surplus GT contributes nothing."""
        valid = np.ones((2, 9), dtype="float32")
        fields = _make_packed(valid=valid, num_targets=9, num_queries=4,
                              batch=2)
        y_true, y_pred = _pack(fields)
        actual = _terms(_loss(), y_true, y_pred)
        assert actual["num_matched"] == 8.0  # 2 images x min(4, 9)
        assert actual["num_boxes"] == 18.0   # all 18 GT still normalize
        expected = _ref_terms(fields)
        for key in ("loss_bbox", "loss_giou", "loss_mask", "loss_dice",
                    "loss_ce", "presence_loss"):
            np.testing.assert_allclose(actual[key], expected[key],
                                       rtol=2e-5, atol=2e-6, err_msg=key)

    def test_an_all_negative_batch_takes_a_finite_gradient_step(self):
        """H-6 end to end: not merely finite VALUES but a finite BACKWARD."""
        zeros = np.zeros((BATCH, TARGETS), dtype="float32")
        fields = _make_packed(valid=zeros)
        y_true, y_pred = _pack(fields)
        tensor = tf.constant(y_pred)
        loss = _loss()
        with tf.GradientTape() as tape:
            tape.watch(tensor)
            value = loss(y_true, tensor)
        grad = ops.convert_to_numpy(tape.gradient(value, tensor))
        assert np.isfinite(float(ops.convert_to_numpy(value)))
        assert np.all(np.isfinite(grad))
        # Liveness: SOMETHING is still learning -- the presence negative.
        assert np.abs(grad[:, QUERIES, PACKED_SCORE_CHANNEL]).max() > 1e-6

    def test_exactly_zero_logits_produce_a_finite_loss_and_gradient(self):
        fields = _make_packed()
        fields["logits"] = np.zeros_like(fields["logits"])
        fields["presence"] = np.zeros_like(fields["presence"])
        fields["pred_masks"] = np.zeros_like(fields["pred_masks"])
        y_true, y_pred = _pack(fields)
        tensor = tf.constant(y_pred)
        loss = _loss()
        with tf.GradientTape() as tape:
            tape.watch(tensor)
            value = loss(y_true, tensor)
        grad = ops.convert_to_numpy(tape.gradient(value, tensor))
        assert np.isfinite(float(ops.convert_to_numpy(value)))
        assert np.all(np.isfinite(grad))
        np.testing.assert_allclose(
            _terms(loss, y_true, y_pred)["loss_ce"],
            _ref_terms(fields)["loss_ce"], rtol=2e-5, atol=2e-6)

    def test_an_unmatched_query_against_a_padded_target_stays_finite(self):
        """The `+ PORT_ONLY(masked-sum)` divergence's exposed corner.

        Unlike the reference, an unmatched query here still evaluates the box
        arithmetic against an all-zero padded GT row. That is finite by the
        same measurement that removed the matcher's dummy-box guard (D-006):
        `union = area_pred + 0 - 0` stays positive. Pinned, not assumed.
        """
        valid = np.array([[1, 0, 0, 0, 0]], dtype="float32")
        fields = _make_packed(valid=valid, batch=1, num_queries=6)
        y_true, y_pred = _pack(fields)
        tensor = tf.constant(y_pred)
        loss = _loss()
        with tf.GradientTape() as tape:
            tape.watch(tensor)
            value = loss(y_true, tensor)
        grad = ops.convert_to_numpy(tape.gradient(value, tensor))
        assert np.all(np.isfinite(grad))
        assert _terms(loss, y_true, y_pred)["num_matched"] == 1.0

    def test_gradient_reaches_every_supervised_channel(self):
        """M3 as a standing check: a term that reaches nothing is a term that
        does nothing, and a falling total would hide it."""
        fields = _make_packed()
        y_true, y_pred = _pack(fields)
        tensor = tf.constant(y_pred)
        with tf.GradientTape() as tape:
            tape.watch(tensor)
            value = _loss()(y_true, tensor)
        grad = ops.convert_to_numpy(tape.gradient(value, tensor))
        assert np.abs(grad[:, :QUERIES, PACKED_SCORE_CHANNEL]).max() > 1e-6
        assert np.abs(
            grad[:, :QUERIES, PACKED_BOX_START:PACKED_MASK_START]).max() > 1e-6
        assert np.abs(grad[:, :QUERIES, PACKED_MASK_START:]).max() > 1e-6
        assert np.abs(grad[:, QUERIES, PACKED_SCORE_CHANNEL]).max() > 1e-6


# --------------------------------------------------------------------------
# Switches: masks, the sqrt(B) multiplier, serialization
# --------------------------------------------------------------------------

class TestSwitches:

    def test_masks_off_gives_exactly_zero_mask_terms_and_a_narrower_tensor(
            self):
        fields = _make_packed()
        y_true, y_pred = _pack(fields, include_masks=False)
        assert y_pred.shape[-1] == PACKED_MASK_START
        actual = _terms(_loss(include_masks=False), y_true, y_pred)
        assert actual["loss_mask"] == 0.0
        assert actual["loss_dice"] == 0.0
        # The four live terms are unchanged by turning masks off.
        wide_true, wide_pred = _pack(fields, include_masks=True)
        wide = _terms(_loss(include_masks=True), wide_true, wide_pred)
        for key in ("loss_ce", "presence_loss", "loss_bbox", "loss_giou"):
            np.testing.assert_allclose(actual[key], wide[key], rtol=1e-5,
                                       atol=1e-7, err_msg=key)

    def test_the_total_excludes_the_mask_weights_when_masks_are_off(self):
        fields = _make_packed()
        y_true, y_pred = _pack(fields, include_masks=False)
        loss = _loss(include_masks=False)
        terms = _terms(loss, y_true, y_pred)
        expected = (20.0 * terms["loss_ce"] + 20.0 * terms["presence_loss"]
                    + 5.0 * terms["loss_bbox"] + 2.0 * terms["loss_giou"])
        np.testing.assert_allclose(
            float(ops.convert_to_numpy(loss(y_true, y_pred))), expected,
            rtol=1e-5, atol=1e-6)

    def test_the_total_is_the_weighted_sum_of_the_six_terms(self):
        fields = _make_packed()
        y_true, y_pred = _pack(fields)
        loss = _loss()
        terms = _terms(loss, y_true, y_pred)
        expected = (20.0 * terms["loss_ce"] + 20.0 * terms["presence_loss"]
                    + 5.0 * terms["loss_bbox"] + 2.0 * terms["loss_giou"]
                    + 200.0 * terms["loss_mask"] + 10.0 * terms["loss_dice"])
        np.testing.assert_allclose(
            float(ops.convert_to_numpy(loss(y_true, y_pred))), expected,
            rtol=1e-5, atol=1e-5)

    def test_scale_by_find_batch_size_multiplies_the_total_by_sqrt_B(self):
        """`sam3_loss.py:192-195`. Implemented, default OFF -- decisions.md
        D-007. The default being off is what this first asserts."""
        fields = _make_packed()
        y_true, y_pred = _pack(fields)
        plain = float(ops.convert_to_numpy(_loss()(y_true, y_pred)))
        scaled = float(ops.convert_to_numpy(
            _loss(scale_by_find_batch_size=True)(y_true, y_pred)))
        np.testing.assert_allclose(scaled, plain * np.sqrt(BATCH), rtol=1e-5)
        assert Sam3DetectionLoss().scale_by_find_batch_size is False

    def test_the_shipped_defaults_are_the_reference_shipped_config(self):
        """M1 as a standing guard: these numbers come from
        `roboflow_v100_full_ft_100_images.yaml`, not from taste."""
        loss = Sam3DetectionLoss()
        assert loss.weight_ce == 20.0
        assert loss.weight_presence == 20.0
        assert loss.weight_bbox == 5.0
        assert loss.weight_giou == 2.0
        assert loss.pos_weight == 10.0
        assert loss.alpha == 0.25
        assert loss.gamma == 2.0
        assert loss.pos_focal is False
        assert loss.weak_loss is False
        assert loss.pad_n_queries == 200
        assert loss.presence_alpha == 0.5
        assert loss.presence_gamma == 0.0
        assert loss.include_masks is False  # decisions.md D-009
        # From the COMMENTED-OUT mask block; no shipped config ran these.
        assert loss.weight_mask == 200.0
        assert loss.weight_dice == 10.0
        assert loss.focal_alpha == 0.25
        assert loss.focal_gamma == 2.0
        # Matcher weights, `roboflow...yaml:180-188`.
        assert (loss.cost_class, loss.cost_bbox, loss.cost_giou) == (
            2.0, 5.0, 2.0)

    @pytest.mark.parametrize("pad", [0, -1, 3.5])
    def test_a_nonsensical_pad_n_queries_raises(self, pad):
        with pytest.raises(ValueError, match="pad_n_queries"):
            Sam3DetectionLoss(pad_n_queries=pad)

    def test_config_round_trip_is_value_exact(self):
        fields = _make_packed()
        y_true, y_pred = _pack(fields)
        original = _loss(weak_loss=True, scale_by_find_batch_size=True,
                         pos_focal=True, gamma=1.5, presence_gamma=0.75,
                         normalize_by_valid_object_num=False)
        config = original.get_config()
        restored = Sam3DetectionLoss.from_config(config)
        assert restored.get_config() == config
        np.testing.assert_allclose(
            float(ops.convert_to_numpy(restored(y_true, y_pred))),
            float(ops.convert_to_numpy(original(y_true, y_pred))),
            rtol=0, atol=0)

    def test_every_constructor_argument_survives_get_config(self):
        """A config that silently drops a field restores a DIFFERENT loss."""
        import inspect
        signature = inspect.signature(Sam3DetectionLoss.__init__)
        expected = {name for name in signature.parameters
                    if name not in ("self", "kwargs")}
        assert expected <= set(Sam3DetectionLoss().get_config())

    def test_the_loss_is_registered_as_serializable(self):
        assert keras.saving.get_registered_name(
            Sam3DetectionLoss) == "Custom>Sam3DetectionLoss"
        assert keras.saving.get_registered_object(
            "Custom>Sam3DetectionLoss") is Sam3DetectionLoss


# --------------------------------------------------------------------------
# M5 -- the loss tolerance measured in BOTH precision regimes
# --------------------------------------------------------------------------

class TestLossPrecisionRegimes:

    @pytest.mark.parametrize("tf32", [True, False])
    def test_every_term_holds_its_tolerance_in_both_tf32_regimes(self, tf32):
        previous = tf.config.experimental.tensor_float_32_execution_enabled()
        tf.config.experimental.enable_tensor_float_32_execution(tf32)
        try:
            fields = _make_packed()
            y_true, y_pred = _pack(fields)
            actual = _terms(_loss(), y_true, y_pred)
            expected = _ref_terms(fields)
            for key in ("loss_ce", "presence_loss", "loss_bbox", "loss_giou",
                        "loss_mask", "loss_dice"):
                np.testing.assert_allclose(actual[key], expected[key],
                                           rtol=2e-5, atol=2e-6,
                                           err_msg=f"{key} @ tf32={tf32}")
        finally:
            tf.config.experimental.enable_tensor_float_32_execution(previous)
