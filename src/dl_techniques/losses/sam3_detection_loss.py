"""Detection supervision for :class:`~dl_techniques.models.sam3.Sam3Image`.

This module currently ships the **Hungarian matcher** only. The loss terms that
consume its assignment (``loss_ce`` / ``presence_loss`` / ``loss_bbox`` /
``loss_giou`` / ``loss_mask`` / ``loss_dice``) land here in a following step;
the matcher is separated because it is the one piece whose feasibility had to be
settled by execution before anything could be designed on top of it.

``import tensorflow`` is SANCTIONED in this module, and only for one reason
-------------------------------------------------------------------------
The assignment itself is :func:`scipy.optimize.linear_sum_assignment`, which has
no ``keras.ops`` equivalent and no differentiable substitute that preserves a
1:1 match. It reaches the graph through :func:`tf.py_function`. That call is
**training-only** -- it runs inside a ``keras.losses.Loss``, never inside any
forward path -- which is exactly the exemption ``src/train/sam/data.py:35``
claims for its ``tf.data`` pipeline. ``import tensorflow`` remains FORBIDDEN in
``src/dl_techniques/models/sam3/*.py``, and every differentiable arithmetic
operation below is written in ``keras.ops``.

The direct consequence is that this loss family **cannot run under XLA**: there
is no ``EagerPyFunc`` kernel for ``XLA_GPU_JIT``, so ``jit_compile=True`` fails
hard rather than degrading. This model family already pins ``jit_compile=False``
independently, so the constraint costs nothing new -- but it is now doubly
binding and any consumer must set it at the ``compile()`` site.

Where the numbers come from
---------------------------
Every cost expression below is transcribed from the pinned upstream clone at
``96914d2425f90a64f45ca977c2b5165418099543``,
``sam3/train/matcher.py::BinaryHungarianMatcherV2.forward`` (the ``focal=True,
stable=False`` branch the one shipped training config selects) and
``sam3/model/box_ops.py``. It is NOT derived from this port's own behaviour.
The divergences from the reference are enumerated, signed and named in
:class:`_Sam3HungarianMatcher`'s docstring rather than left to be rediscovered.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import tensorflow as tf
from keras import ops
from scipy.optimize import linear_sum_assignment

# The reference's own sentinel pair: an entry excluded from matching is costed
# `INVALID_COST` and every produced pair whose cost is at or above
# `VALID_COST_THRESHOLD` is dropped afterwards (`matcher.py:16-26, 608-614`).
# They are deliberately three orders apart so a legitimately large real cost can
# never be mistaken for a sentinel.
INVALID_COST: float = 1e9
VALID_COST_THRESHOLD: float = 1e8


def box_cxcywh_to_xyxy(boxes: Any) -> Any:
    """Convert boxes from ``cxcywh`` to ``xyxy``.

    Interface contract: ``boxes`` is a tensor whose LAST axis has size 4 and
    carries ``(center_x, center_y, width, height)``; the return has the same
    shape and dtype and carries ``(x0, y0, x1, y1)``. Leading axes are
    untouched, so the same call serves ``(N, 4)`` and ``(B, Q, 4)``. It never
    raises and never validates -- a negative width simply produces ``x1 < x0``,
    which the GIoU below then reads as a zero-area box.

    Transcribed from ``sam3/model/box_ops.py::box_cxcywh_to_xyxy``::

        [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]

    This conversion lives here because ``models/sam3/`` deliberately does not
    ship it: ``pred_boxes`` is normalized ``cxcywh`` by contract and the package
    treats the ``xyxy`` form as a consumer-side concern
    (``sam3_image.py:73``). This is that consumer.

    :param boxes: Boxes in ``cxcywh``, last axis 4.
    :type boxes: Any
    :return: The same boxes in ``xyxy``.
    :rtype: Any
    """
    center_x = boxes[..., 0]
    center_y = boxes[..., 1]
    half_width = 0.5 * boxes[..., 2]
    half_height = 0.5 * boxes[..., 3]
    return ops.stack(
        [center_x - half_width, center_y - half_height,
         center_x + half_width, center_y + half_height],
        axis=-1,
    )


def pairwise_generalized_iou(boxes_a: Any, boxes_b: Any) -> Any:
    """Generalized IoU between every pair drawn from two batched box sets.

    Interface contract: ``boxes_a`` is ``(B, N, 4)`` and ``boxes_b`` is
    ``(B, M, 4)``, both in ``xyxy``; the return is ``(B, N, M)`` with
    ``result[b, n, m] = GIoU(boxes_a[b, n], boxes_b[b, m])``, in the same dtype.
    A degenerate box (zero area on BOTH sides of a pair) makes both the union
    and the enclosing area zero and the result is therefore ``nan`` -- callers
    that can produce degenerate boxes must exclude them, which
    :class:`_Sam3HungarianMatcher` does by substitution before it ever calls
    this. It never raises.

    Transcribed term by term from ``sam3/model/box_ops.py``::

        area   = (x1 - x0) * (y1 - y0)                        # box_area
        lt     = max(a[..., None, :2], b[..., None, :, :2])   # box_iou
        rb     = min(a[..., None, 2:], b[..., None, :, 2:])
        inter  = clamp(rb - lt, min=0).prod(-1)
        union  = area_a[..., None] + area_b[..., None, :] - inter
        iou    = inter / union
        lt_e   = min(a[..., None, :2], b[..., None, :, :2])   # generalized_box_iou
        rb_e   = max(a[..., None, 2:], b[..., None, :, 2:])
        enclose= clamp(rb_e - lt_e, min=0).prod(-1)
        giou   = iou - (enclose - union) / enclose

    Note the ``min``/``max`` swap between the intersection rectangle and the
    enclosing rectangle -- it is the whole difference between IoU and GIoU and
    is easy to transcribe backwards.

    :param boxes_a: ``(B, N, 4)`` boxes in ``xyxy``.
    :type boxes_a: Any
    :param boxes_b: ``(B, M, 4)`` boxes in ``xyxy``.
    :type boxes_b: Any
    :return: ``(B, N, M)`` generalized IoU.
    :rtype: Any
    """
    area_a = (boxes_a[..., 2] - boxes_a[..., 0]) * (
        boxes_a[..., 3] - boxes_a[..., 1])
    area_b = (boxes_b[..., 2] - boxes_b[..., 0]) * (
        boxes_b[..., 3] - boxes_b[..., 1])

    top_left_a = boxes_a[:, :, None, :2]
    top_left_b = boxes_b[:, None, :, :2]
    bottom_right_a = boxes_a[:, :, None, 2:]
    bottom_right_b = boxes_b[:, None, :, 2:]

    inter_wh = ops.maximum(
        ops.minimum(bottom_right_a, bottom_right_b)
        - ops.maximum(top_left_a, top_left_b),
        0.0,
    )
    intersection = inter_wh[..., 0] * inter_wh[..., 1]
    union = area_a[:, :, None] + area_b[:, None, :] - intersection
    iou = intersection / union

    enclose_wh = ops.maximum(
        ops.maximum(bottom_right_a, bottom_right_b)
        - ops.minimum(top_left_a, top_left_b),
        0.0,
    )
    enclose = enclose_wh[..., 0] * enclose_wh[..., 1]
    return iou - (enclose - union) / enclose


def _solve_assignments(
        cost: "np.ndarray") -> Tuple["np.ndarray", "np.ndarray"]:
    """Run ``linear_sum_assignment`` per batch element and drop sentinel pairs.

    Interface contract: ``cost`` is a finite ``(B, Q, N)`` float array; the
    return is ``(assignment, is_matched)``, both ``(B, Q)`` -- ``assignment``
    is ``int32`` giving the matched target column for each query (``0`` where
    unmatched, which the caller must not read without consulting the mask) and
    ``is_matched`` is ``float32``, ``1.0`` exactly where a pair survived the
    ``< VALID_COST_THRESHOLD`` filter. It never raises for any shape including
    ``N == 0``; a non-finite cost would propagate into ``scipy``, which is why
    the caller sanitizes.

    This is the numpy body of the ``tf.py_function``. It is a module-level
    function, not a closure, so it is importable by a test directly.

    :param cost: ``(B, Q, N)`` cost matrix.
    :type cost: np.ndarray
    :return: ``(assignment, is_matched)``, each ``(B, Q)``.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    cost = np.asarray(cost, dtype=np.float64)
    batch, num_queries, _ = cost.shape
    assignment = np.zeros((batch, num_queries), dtype=np.int32)
    is_matched = np.zeros((batch, num_queries), dtype=np.float32)
    for index in range(batch):
        rows, columns = linear_sum_assignment(cost[index])
        if rows.size == 0:
            continue
        keep = cost[index][rows, columns] < VALID_COST_THRESHOLD
        rows, columns = rows[keep], columns[keep]
        assignment[index, rows] = columns.astype(np.int32)
        is_matched[index, rows] = 1.0
    return assignment, is_matched


class _Sam3HungarianMatcher:
    """One-to-one Hungarian assignment between SAM 3 queries and padded GT.

    Reference: ``sam3/train/matcher.py::BinaryHungarianMatcherV2.forward`` at
    the pinned clone, ``focal=True, stable=False`` -- the branch the one shipped
    training config (``roboflow_v100_full_ft_100_images.yaml:180-188``) selects.
    The cost is::

        prob        = sigmoid(logits)                                # (B, Q)
        cost_class  = -alpha * (1 - prob)**gamma * logsigmoid( logits)
                      + (1 - alpha) * prob**gamma * logsigmoid(-logits)
        cost_bbox   = cdist(pred_cxcywh, gt_cxcywh, p=1)             # (B, Q, N)
        cost_giou   = -GIoU(xyxy(pred), xyxy(gt))                    # (B, Q, N)
        C = cost_bbox_weight * cost_bbox
          + cost_class_weight * cost_class[..., None]
          + cost_giou_weight * cost_giou

    ``cost_class`` is a per-QUERY scalar broadcast across targets (the
    reference's ``.unsqueeze(-1).expand_as(cost_bbox)``); SAM 3 is a binary
    grounding detector, so there is no per-target class column to index.

    Divergences from the reference, SIGNED and NAMED
    ------------------------------------------------
    ``this_port = reference + PORT_ONLY(...) - REFERENCE_ONLY(...)``

    * ``+ PORT_ONLY(always-on padded validity)``. The reference slices each
      image's cost to its own ``num_boxes`` (``C[i, :, :s]``) and only engages
      the ``1e9``/``<1e8`` machinery when a validity mask is supplied. This port
      has a single fixed ``N_max`` contract and ALWAYS supplies one, so the
      sentinel path is always live. The two agree by construction: the columns
      the reference slices away are exactly the columns this port costs
      ``1e9`` and then filters, and both produce ``min(Q, num_valid)`` pairs.
    * ``- REFERENCE_ONLY(out_is_valid)``. The reference accepts a per-QUERY
      validity mask for the o2m/aux batched paths. Every one of this model's
      ``Q`` queries is always real, so the parameter is not ported rather than
      ported-and-always-True.
    * ``- REFERENCE_ONLY(repeats / repeat_batch)``. Target tiling for the DAC
      one-to-many branch and for batched auxiliary outputs. DAC query doubling
      is deferred wholesale (with its own non-Hungarian greedy matcher), so its
      tiling argument is deferred with it.
    * ``- REFERENCE_ONLY(remove_samples_with_0_gt)``. The reference drops
      zero-GT images from the batch before matching and re-indexes afterwards.
      Here a zero-GT image has every target invalid, so every cost is ``1e9``,
      every pair is filtered, and the image contributes no matches -- the same
      outcome without the re-indexing bookkeeping, and it keeps the returned
      tensors statically shaped ``(B, Q)``, which a graph-mode ``Loss`` needs.
    * ``- REFERENCE_ONLY(stable=True branch)`` and
      ``- REFERENCE_ONLY(focal=False branch)``. Neither is selected by any
      shipped config. They are not ported as dead switches.
    * ``- REFERENCE_ONLY(@torch.no_grad)``. Not needed: the only values leaving
      this object are integer indices and a 0/1 mask, neither of which carries a
      gradient, and the assignment crosses a ``tf.py_function`` boundary that is
      non-differentiable regardless.

    **Masks are never costed** -- not by this class and not by any of the five
    matcher classes in the reference, even when segmentation is enabled.

    This class is module-private on purpose. It has exactly one production
    consumer (the detection loss in this same module) and the earned-abstraction
    rule says a single-call-site helper does not deserve public API status or
    the serialization surface that comes with it; its hyperparameters are owned
    and round-tripped by that loss instead. Tests import it by its private name,
    which does not make it public.

    :param cost_class: Weight on the classification cost. Shipped ``2.0``.
    :type cost_class: float
    :param cost_bbox: Weight on the L1 box cost. Shipped ``5.0``.
    :type cost_bbox: float
    :param cost_giou: Weight on the negated-GIoU cost. Shipped ``2.0``.
    :type cost_giou: float
    :param alpha: Focal balance in the classification cost. Shipped ``0.25``.
    :type alpha: float
    :param gamma: Focal focusing exponent. Shipped ``2.0``.
    :type gamma: float
    :raises ValueError: If all three cost weights are zero, which would make
        every assignment arbitrary (the reference asserts the same thing).
    """

    def __init__(
            self,
            cost_class: float = 2.0,
            cost_bbox: float = 5.0,
            cost_giou: float = 2.0,
            alpha: float = 0.25,
            gamma: float = 2.0,
    ) -> None:
        if cost_class == 0.0 and cost_bbox == 0.0 and cost_giou == 0.0:
            raise ValueError(
                "_Sam3HungarianMatcher: all three cost weights are zero, so "
                "every assignment would be arbitrary. Set at least one of "
                "cost_class / cost_bbox / cost_giou to a non-zero value.")
        self.cost_class = float(cost_class)
        self.cost_bbox = float(cost_bbox)
        self.cost_giou = float(cost_giou)
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor arguments.

        Not a Keras serialization hook -- this class is not registered and is
        not a ``Layer``. It exists so the owning loss can round-trip the
        matcher's hyperparameters through its own ``get_config()``.

        :return: The five cost/focal hyperparameters.
        :rtype: Dict[str, Any]
        """
        return {
            "cost_class": self.cost_class,
            "cost_bbox": self.cost_bbox,
            "cost_giou": self.cost_giou,
            "alpha": self.alpha,
            "gamma": self.gamma,
        }

    def cost_matrix(
            self,
            pred_logits: Any,
            pred_boxes: Any,
            target_boxes: Any,
            target_valid: Any,
    ) -> Any:
        """Build the ``(B, Q, N)`` matching cost.

        Interface contract: all four inputs share a batch axis ``B``;
        ``pred_logits`` is ``(B, Q, 1)`` or ``(B, Q)`` raw logits,
        ``pred_boxes`` is ``(B, Q, 4)`` normalized ``cxcywh``, ``target_boxes``
        is ``(B, N, 4)`` normalized ``cxcywh`` padded to a fixed ``N``, and
        ``target_valid`` is ``(B, N)`` with ``1`` at a real GT row and ``0`` at
        padding. The return is ``(B, Q, N)``, finite everywhere, with every
        column of an invalid target set to exactly ``INVALID_COST``. Nothing is
        reduced and no gradient is intended to flow through it.

        :param pred_logits: ``(B, Q, 1)`` or ``(B, Q)`` class logits.
        :type pred_logits: Any
        :param pred_boxes: ``(B, Q, 4)`` predicted boxes, ``cxcywh``.
        :type pred_boxes: Any
        :param target_boxes: ``(B, N, 4)`` padded GT boxes, ``cxcywh``.
        :type target_boxes: Any
        :param target_valid: ``(B, N)`` GT validity, 1 real / 0 padding.
        :type target_valid: Any
        :return: ``(B, Q, N)`` cost matrix.
        :rtype: Any
        """
        pred_boxes = ops.cast(pred_boxes, "float32")
        target_boxes = ops.cast(target_boxes, "float32")
        scores = ops.cast(pred_logits, "float32")
        if len(scores.shape) == 3:
            scores = ops.squeeze(scores, axis=-1)
        valid = ops.cast(target_valid, "float32")

        # DECISION plan-2026-08-05T124709-6c4fac48/D-006
        # There is deliberately NO dummy-box substitution for padded target
        # rows here, and this is the second version of this code: the first one
        # had one, on the stated grounds that an all-zero padded row makes the
        # GIoU `0/0`. That claim is FALSE and was falsified by measurement.
        # `union = area_pred + area_target - inter`, so an all-zero TARGET row
        # still leaves `union = area_pred > 0` and the GIoU comes out finite
        # (measured -0.75 on the probe case, not nan). A `nan` needs BOTH boxes
        # degenerate, and `pred_boxes` is `sigmoid`-bounded, so its width is
        # zero only on a float32-underflowed logit -- and even then the pair is
        # in a padded COLUMN, which the `ops.where` below overwrites with the
        # sentinel. The substitution was mutation-tested and came back INERT on
        # BOTH the value and the gradient path. Do NOT re-add it: it is
        # unreachable defensive code justified by arithmetic that does not
        # hold. `test_cost_is_finite_with_a_degenerate_prediction_and_padding`
        # pins the actual nan-producing configuration instead.
        # See decisions.md D-006.

        # logsigmoid(x) = -softplus(-x). Spelled through `softplus` and never
        # through `log(sigmoid(x))`, which underflows to -inf on a saturated
        # logit; `sigmoid` itself is still needed for the focal modulator, but
        # only inside a bounded power, never inside a log.
        probability = ops.sigmoid(scores)
        log_probability = -ops.softplus(-scores)
        log_one_minus_probability = -ops.softplus(scores)
        cost_class = (
            -self.alpha * ops.power(1.0 - probability, self.gamma)
            * log_probability
            + (1.0 - self.alpha) * ops.power(probability, self.gamma)
            * log_one_minus_probability
        )

        cost_bbox = ops.sum(
            ops.abs(pred_boxes[:, :, None, :] - target_boxes[:, None, :, :]),
            axis=-1,
        )
        cost_giou = -pairwise_generalized_iou(
            box_cxcywh_to_xyxy(pred_boxes), box_cxcywh_to_xyxy(target_boxes))

        cost = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class[:, :, None]
            + self.cost_giou * cost_giou
        )
        return ops.where(valid[:, None, :] > 0.0, cost,
                         ops.full_like(cost, INVALID_COST))

    def __call__(
            self,
            pred_logits: Any,
            pred_boxes: Any,
            target_boxes: Any,
            target_valid: Any,
    ) -> Tuple[Any, Any]:
        """Assign each query to at most one valid target.

        Interface contract: inputs are exactly :meth:`cost_matrix`'s. The return
        is ``(assignment, is_matched)``, both ``(B, Q)``: ``assignment`` is
        ``int32``, the matched target index, and is meaningless (``0``) wherever
        ``is_matched`` is ``0.0``; ``is_matched`` is ``float32`` in ``{0, 1}``
        and sums per image to ``min(Q, number of valid targets)``. Neither
        output carries a gradient. It never raises, including for an image with
        no valid target at all, which simply yields an all-zero ``is_matched``
        row.

        :param pred_logits: ``(B, Q, 1)`` or ``(B, Q)`` class logits.
        :type pred_logits: Any
        :param pred_boxes: ``(B, Q, 4)`` predicted boxes, ``cxcywh``.
        :type pred_boxes: Any
        :param target_boxes: ``(B, N, 4)`` padded GT boxes, ``cxcywh``.
        :type target_boxes: Any
        :param target_valid: ``(B, N)`` GT validity, 1 real / 0 padding.
        :type target_valid: Any
        :return: ``(assignment, is_matched)``, each ``(B, Q)``.
        :rtype: Tuple[Any, Any]
        """
        cost = self.cost_matrix(pred_logits, pred_boxes, target_boxes,
                                target_valid)
        cost = ops.stop_gradient(cost)
        assignment, is_matched = tf.py_function(
            func=_solve_assignments,
            inp=[cost],
            Tout=[tf.int32, tf.float32],
        )
        # `py_function` erases static shape. Restore it: the graph-mode `Loss`
        # downstream needs `(B, Q)` to build its gathers.
        static = cost.shape[:2]
        assignment.set_shape(static)
        is_matched.set_shape(static)
        return assignment, is_matched


__all__: List[str] = [
    "INVALID_COST",
    "VALID_COST_THRESHOLD",
    "box_cxcywh_to_xyxy",
    "pairwise_generalized_iou",
]
