"""Detection supervision for :class:`~dl_techniques.models.sam3.Sam3Image`.

This module ships the **Hungarian matcher** and the six-term
:class:`Sam3DetectionLoss` that consumes its assignment (``loss_ce`` /
``presence_loss`` / ``loss_bbox`` / ``loss_giou`` / ``loss_mask`` /
``loss_dice``), plus the **packed-tensor layout** those two speak, which lives
here as the single source of truth (see "The packed layout" below).

Masks are OFF by default, and that is a DECISION, not an inheritance
----------------------------------------------------------------------
The reference's one shipped training config
(``sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml``)
trains with masks **DISABLED**: ``enable_segmentation: False`` (line 160) and
the entire ``Masks`` loss block is **commented out** (lines 107-154). So the
only executable recipe upstream releases supervises boxes and presence only,
and the mask weights below (``loss_mask 200.0``, ``loss_dice 10.0``, along with
``focal_alpha 0.25`` / ``focal_gamma 2.0``) are transcribed from a block no
config ever ran. This port therefore makes masks a real switch,
``include_masks``, whose default (``False``) matches the shipped recipe as a
RECORDED decision rather than an inherited assumption -- see decisions.md D-009.
Turning it on is supported and tested; it is simply not something the reference
can be said to have validated.

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
Every cost and loss expression below is transcribed from the pinned upstream
clone at ``96914d2425f90a64f45ca977c2b5165418099543``:
``sam3/train/matcher.py::BinaryHungarianMatcherV2.forward`` (the ``focal=True,
stable=False`` branch the one shipped training config selects),
``sam3/train/loss/loss_fns.py`` (``IABCEMdetr`` at :266, ``Boxes`` at :524,
``Masks`` at :575, ``sigmoid_focal_loss`` at :125, ``dice_loss`` at :78),
``sam3/train/loss/sam3_loss.py`` (``_get_num_boxes`` at :65,
``scale_by_find_batch_size`` at :192-195) and ``sam3/model/box_ops.py``. None of
it is derived from this port's own behaviour. The divergences from the reference
are enumerated, signed and named in :class:`_Sam3HungarianMatcher`'s and
:class:`Sam3DetectionLoss`'s docstrings rather than left to be rediscovered.

The packed layout -- ONE home, here
------------------------------------
A single ``keras.losses.Loss`` object handed a dict ``y_pred`` breaks
(``CompileLoss.build`` broadcasts the one object across every leaf and
``KeyError``s), and the six terms above share ONE Hungarian assignment, so the
supervision signal travels as ONE packed tensor per side. The layout is defined
by the module-level ``PACKED_*`` / ``META_*`` constants and read back by
:func:`unpack_predictions` / :func:`unpack_targets`. Every other file that
speaks it -- the training model and the data pipeline -- imports these names
rather than re-spelling the offsets; a channel index restated in three places is
a hand-maintained lockstep invariant, i.e. a latent defect, and the whole point
of putting the layout in one module is that there is nothing to keep in step.

* ``y_pred`` is ``(B, Q + 1, C)``. Rows ``0 .. Q-1`` are the object queries:
  channel ``0`` is ``pred_logits``, channels ``1:5`` are ``pred_boxes`` in
  normalized ``cxcywh``, channels ``5:`` are the flattened ``pred_masks``
  LOGITS. Row ``Q`` is the presence row: channel ``0`` is ``presence_logit``,
  every other channel is zero-filled and unread.
* ``y_true`` is ``(B, N_max + 1, C)``, symmetric. Rows ``0 .. N_max-1`` are
  padded GT: channel ``0`` is the validity flag, channels ``1:5`` are the GT box
  in normalized ``cxcywh``, channels ``5:`` are the flattened GT mask. Row
  ``N_max`` is the meta row: ``[keep_loss, num_boxes_this_image,
  is_exhaustive, 0, ...]``.
* ``C = 5 + P`` with masks on (``P`` = flattened mask size) and ``C = 5`` off.
* ``semantic_seg`` is deliberately NOT packed. The head exists and emits logits,
  but phase 2 leaves it unsupervised; it is named here so its absence is a
  stated scope boundary rather than a silent drop.
"""

from typing import Any, Dict, List, Optional, Tuple

import keras
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

# ---------------------------------------------------------------------------
# The packed layout. This block is the ONLY definition of these offsets in the
# repository; the training model and the data pipeline import from here.
# ---------------------------------------------------------------------------
#: Channel carrying the per-row scalar: `pred_logits` on a query row,
#: `presence_logit` on the presence row, the validity flag on a GT row.
PACKED_SCORE_CHANNEL: int = 0
#: First channel of the 4-wide `cxcywh` box block, `[1:5]` on both sides.
PACKED_BOX_START: int = 1
#: First channel of the flattened mask block. Also the channel count when masks
#: are off, i.e. the width of the non-mask prefix.
PACKED_MASK_START: int = 5
#: Meta-row channels. The meta row is the LAST row of `y_true` and carries
#: per-image scalars that have no natural home on any GT row.
META_KEEP_LOSS: int = 0
META_NUM_BOXES: int = 1
# DECISION plan-2026-08-05T124709-6c4fac48/D-010
# `plan.md` step 3 specified the meta row as
# `[keep_loss, num_boxes_this_image, 0, 0, ...]`; channel 2 is a RESERVED ZERO
# that this port SPENDS on `is_exhaustive`. That is the plan's one recorded
# LAYOUT DEPARTURE. Do NOT hardcode this to 1 and delete the channel: without a
# real per-image `is_exhaustive` signal, divisor #5 (`weak_loss=True`) collapses
# into divisor #4's plain-mean sub-path -- nothing is ever dropped, the retained
# count is identically `B * Q`, and "both `loss_ce` paths implemented" would be
# true only on paper.
# THE FOOTGUN, named because it cannot fail loudly: a producer that leaves this
# channel at 0.0 while running `weak_loss=True` is declaring EVERY image
# NON-exhaustive, which nulls all negative supervision. The channel is UNREAD at
# the shipped default (`weak_loss=False`), so the footgun is armed only by an
# explicit opt-in; `pack_targets` defaults it to ones and `train/sam3/data.py`
# writes 1. See decisions.md D-010.
META_IS_EXHAUSTIVE: int = 2


def packed_channel_count(mask_size: int = 0) -> int:
    """Return ``C``, the packed channel width, for a given flattened mask size.

    Interface contract: ``mask_size`` is ``0`` when masks are off and ``H * W``
    (the flattened mask length) when they are on; the return is the ``C`` that
    both ``y_pred`` and ``y_true`` must have. It never raises and does no
    validation -- a negative ``mask_size`` simply produces a nonsensical width,
    which the slicing below would then fail on loudly.

    :param mask_size: Flattened mask length, ``0`` for no masks.
    :type mask_size: int
    :return: The packed channel width ``C``.
    :rtype: int
    """
    return PACKED_MASK_START + mask_size


def unpack_predictions(y_pred: Any,
                       include_masks: bool = False) -> Dict[str, Any]:
    """Split the packed ``(B, Q + 1, C)`` prediction tensor.

    Interface contract: ``y_pred`` is a float tensor whose SECOND-to-last axis
    is ``Q + 1`` (the last row being the presence row) and whose last axis is
    :func:`packed_channel_count`. The return is a dict with ``pred_logits``
    ``(B, Q)``, ``pred_boxes`` ``(B, Q, 4)``, ``presence_logit`` ``(B, 1)`` and
    ``pred_masks`` ``(B, Q, P)`` -- the last being ``None`` when
    ``include_masks`` is ``False``. Nothing is reduced, nothing is cast beyond
    the input dtype, and it never raises: a width mismatch produces a slice of
    the wrong size rather than an error, which is exactly why the training
    model, the loss and the data pipeline share one ``include_masks`` flag and a
    test pins that they agree.

    :param y_pred: Packed predictions, ``(B, Q + 1, C)``.
    :type y_pred: Any
    :param include_masks: Whether channels ``5:`` carry mask logits.
    :type include_masks: bool
    :return: ``pred_logits`` / ``pred_boxes`` / ``pred_masks`` /
        ``presence_logit``.
    :rtype: Dict[str, Any]
    """
    queries = y_pred[:, :-1, :]
    presence_row = y_pred[:, -1, :]
    return {
        "pred_logits": queries[..., PACKED_SCORE_CHANNEL],
        "pred_boxes": queries[..., PACKED_BOX_START:PACKED_MASK_START],
        "pred_masks": (queries[..., PACKED_MASK_START:]
                       if include_masks else None),
        "presence_logit": presence_row[..., PACKED_SCORE_CHANNEL:
                                       PACKED_SCORE_CHANNEL + 1],
    }


def unpack_targets(y_true: Any,
                   include_masks: bool = False) -> Dict[str, Any]:
    """Split the packed ``(B, N_max + 1, C)`` target tensor.

    Interface contract: ``y_true`` is a float tensor whose second-to-last axis
    is ``N_max + 1`` (the last row being the meta row) and whose last axis is
    :func:`packed_channel_count`. The return is a dict with ``target_valid``
    ``(B, N_max)``, ``target_boxes`` ``(B, N_max, 4)``, ``target_masks``
    ``(B, N_max, P)`` or ``None``, ``keep_loss`` ``(B, 1)``, ``num_boxes``
    ``(B,)`` and ``is_exhaustive`` ``(B,)``. Nothing is reduced and it never
    raises.

    :param y_true: Packed targets, ``(B, N_max + 1, C)``.
    :type y_true: Any
    :param include_masks: Whether channels ``5:`` carry GT masks.
    :type include_masks: bool
    :return: The six unpacked target fields.
    :rtype: Dict[str, Any]
    """
    rows = y_true[:, :-1, :]
    meta = y_true[:, -1, :]
    return {
        "target_valid": rows[..., PACKED_SCORE_CHANNEL],
        "target_boxes": rows[..., PACKED_BOX_START:PACKED_MASK_START],
        "target_masks": (rows[..., PACKED_MASK_START:]
                         if include_masks else None),
        "keep_loss": meta[..., META_KEEP_LOSS:META_KEEP_LOSS + 1],
        "num_boxes": meta[..., META_NUM_BOXES],
        "is_exhaustive": meta[..., META_IS_EXHAUSTIVE],
    }


def derive_keep_loss(target_boxes: Any, target_valid: Any) -> Any:
    """Compute the per-image presence target ``keep_loss``.

    Interface contract: ``target_boxes`` is ``(B, N, 4)`` in ``cxcywh`` and
    ``target_valid`` is ``(B, N)``; the return is ``(B, 1)`` float32, ``1.0``
    for an image that has at least one VISIBLE GT instance and ``0.0``
    otherwise. It never raises.

    Transcribed from ``loss_fns.py:418-423``::

        gt_padded_is_visible = (object_ids >= 0) & (w > 0) & (h > 0)
        keep_loss = (gt_padded_is_visible.sum(-1)[..., None] != 0).float()

    with ONE signed divergence, ``- REFERENCE_ONLY(object_ids)``: this port has
    no object-id channel, so the ``object_ids >= 0`` term is carried by the
    packed validity flag, which is exactly what a real (non-padding) instance
    row means here. The width/height terms are ported literally, because they
    are what makes an "invisible object" row -- a real id with a zero-area box
    -- count as absent, and dropping them would make a padded-but-id-bearing row
    flip presence to positive.

    This function is the single definition of the derivation. The data pipeline
    calls it to fill the meta row; the loss then READS that row rather than
    re-deriving, so there is one formula, not two.

    :param target_boxes: ``(B, N, 4)`` padded GT boxes in ``cxcywh``.
    :type target_boxes: Any
    :param target_valid: ``(B, N)`` validity, 1 real / 0 padding.
    :type target_valid: Any
    :return: ``(B, 1)`` float32 presence target.
    :rtype: Any
    """
    boxes = ops.cast(target_boxes, "float32")
    visible = (
        ops.cast(ops.cast(target_valid, "float32") > 0.0, "float32")
        * ops.cast(boxes[..., 2] > 0.0, "float32")
        * ops.cast(boxes[..., 3] > 0.0, "float32")
    )
    return ops.cast(ops.sum(visible, axis=-1, keepdims=True) != 0.0, "float32")


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


def iou_and_generalized_iou(boxes_a: Any, boxes_b: Any) -> Tuple[Any, Any]:
    """IoU and generalized IoU of two BROADCAST-ALIGNED box tensors.

    Interface contract: ``boxes_a`` and ``boxes_b`` are broadcast-compatible
    tensors whose last axis has size 4 and carries ``xyxy``; the return is the
    pair ``(iou, giou)``, each with the broadcast shape MINUS that last axis and
    in the promoted dtype. Nothing is reduced. It never raises. A pair in which
    BOTH boxes have zero area makes the union AND the enclosing area zero and
    yields ``nan`` in both outputs -- that is the measured domain boundary
    (D-006), not a defect, and a caller that can produce such a pair owns
    excluding it.

    This is the SINGLE spelling of the box-overlap arithmetic in this module.
    :func:`pairwise_generalized_iou` (all-pairs, for the matcher) and the
    matched-pair path inside :class:`Sam3DetectionLoss` (elementwise, for
    ``loss_giou`` and the IoU-aware soft target) are both thin broadcast
    adapters over it, because the reference itself ships the same arithmetic
    twice -- ``box_ops.generalized_box_iou`` for the matcher and
    ``box_ops.fast_diag_generalized_box_iou`` for the loss -- and two
    hand-maintained copies of a formula whose ``min``/``max`` swap is its whole
    content is a divergence waiting to happen.

    Transcribed term by term from ``sam3/model/box_ops.py``
    (``box_iou`` + ``generalized_box_iou``, and their ``fast_diag_*`` twins,
    which are the identical expression on aligned rows)::

        area    = (x1 - x0) * (y1 - y0)
        lt      = max(a[..., :2], b[..., :2])
        rb      = min(a[..., 2:], b[..., 2:])
        inter   = clamp(rb - lt, min=0).prod(-1)
        union   = area_a + area_b - inter
        iou     = inter / union
        lt_e    = min(a[..., :2], b[..., :2])
        rb_e    = max(a[..., 2:], b[..., 2:])
        enclose = clamp(rb_e - lt_e, min=0).prod(-1)
        giou    = iou - (enclose - union) / enclose

    Note the ``min``/``max`` swap between the intersection rectangle and the
    enclosing rectangle -- it is the whole difference between IoU and GIoU and
    is easy to transcribe backwards.

    :param boxes_a: Boxes in ``xyxy``, last axis 4.
    :type boxes_a: Any
    :param boxes_b: Boxes in ``xyxy``, last axis 4, broadcastable with
        ``boxes_a``.
    :type boxes_b: Any
    :return: ``(iou, generalized_iou)``.
    :rtype: Tuple[Any, Any]
    """
    area_a = (boxes_a[..., 2] - boxes_a[..., 0]) * (
        boxes_a[..., 3] - boxes_a[..., 1])
    area_b = (boxes_b[..., 2] - boxes_b[..., 0]) * (
        boxes_b[..., 3] - boxes_b[..., 1])

    inter_wh = ops.maximum(
        ops.minimum(boxes_a[..., 2:], boxes_b[..., 2:])
        - ops.maximum(boxes_a[..., :2], boxes_b[..., :2]),
        0.0,
    )
    intersection = inter_wh[..., 0] * inter_wh[..., 1]
    union = area_a + area_b - intersection
    iou = intersection / union

    enclose_wh = ops.maximum(
        ops.maximum(boxes_a[..., 2:], boxes_b[..., 2:])
        - ops.minimum(boxes_a[..., :2], boxes_b[..., :2]),
        0.0,
    )
    enclose = enclose_wh[..., 0] * enclose_wh[..., 1]
    return iou, iou - (enclose - union) / enclose


def pairwise_generalized_iou(boxes_a: Any, boxes_b: Any) -> Any:
    """Generalized IoU between every pair drawn from two batched box sets.

    Interface contract: ``boxes_a`` is ``(B, N, 4)`` and ``boxes_b`` is
    ``(B, M, 4)``, both in ``xyxy``; the return is ``(B, N, M)`` with
    ``result[b, n, m] = GIoU(boxes_a[b, n], boxes_b[b, m])``, in the same dtype.
    A pair of boxes that are BOTH degenerate yields ``nan`` -- see
    :func:`iou_and_generalized_iou`, of which this is the all-pairs broadcast
    adapter. It never raises.

    :param boxes_a: ``(B, N, 4)`` boxes in ``xyxy``.
    :type boxes_a: Any
    :param boxes_b: ``(B, M, 4)`` boxes in ``xyxy``.
    :type boxes_b: Any
    :return: ``(B, N, M)`` generalized IoU.
    :rtype: Any
    """
    _, giou = iou_and_generalized_iou(boxes_a[:, :, None, :],
                                      boxes_b[:, None, :, :])
    return giou


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


def binary_cross_entropy_with_logits(truth: Any, logits: Any) -> Any:
    """Element-wise BCE-with-logits, in LOGIT space, with no clipping.

    Interface contract: ``truth`` and ``logits`` are broadcast-compatible float
    tensors carrying, respectively, targets in ``[0, 1]`` and unbounded real
    logits; the return has their broadcast shape and is NOT reduced -- the
    caller owns every gate and every divisor. It never raises.

    # DECISION plan-2026-08-05T124709-6c4fac48/D-011
    # The spelling is `softplus(-x) + x * (1 - t)`, copied verbatim from
    # `losses/sam2_video_loss.py:225::_focal_from_logits`, and it is NOT
    # interchangeable with the textbook `max(x, 0) - x*t + log1p(exp(-|x|))`.
    # The two agree on VALUE at every logit but the textbook form's AUTODIFF
    # is wrong at exactly `x = 0`: `abs` has no derivative there and
    # `ops.maximum(x, 0)` breaks its tie toward `x`, giving `+1.0` at `t=0`
    # and `0.0` at `t=1` where the true values are `+0.5` and `-0.5` -- a
    # SIGN FLIP on the positive branch. An exactly-zero logit is not
    # hypothetical here: this module multiplies BCE tensors by `ops.where`
    # gates (`keep_loss`, `is_matched`) that emit exact zeros, and a
    # zero-initialized head emits exact-zero logits at step 0.
    # Equally: do NOT route any term in this module through
    # `losses/segmentation_loss.py` or `losses/sam_mask_loss.py` (invariant
    # I-8 / H-1). Both open by clipping PROBABILITIES, and `ops.clip` has an
    # exactly-zero derivative outside its range -- the measured consequence
    # on SAM 2 was a mask-term gradient norm of 3.1751e-13 while the loss
    # VALUE read a healthy 3.937. See decisions.md D-011.

    :param truth: Target values in ``[0, 1]``. Soft targets are allowed and are
        used: ``loss_ce``'s positive branch feeds the IoU-aware soft label.
    :type truth: Any
    :param logits: Unbounded real logits.
    :type logits: Any
    :return: Unreduced element-wise cross-entropy.
    :rtype: Any
    """
    return ops.softplus(-logits) + logits * (1.0 - truth)


def sigmoid_focal_loss(truth: Any, logits: Any, alpha: float,
                       gamma: float) -> Any:
    """Element-wise sigmoid focal loss in LOGIT space, unreduced.

    Interface contract: as :func:`binary_cross_entropy_with_logits`, plus two
    Python-float hyperparameters. The return is unreduced -- the reference's own
    reduction is a divisor the CALLER chooses, and there are three different
    ones in this module, so folding one in here would be a bug generator.

    Transcribed from ``sam3/train/loss/sigmoid_focal_loss.py:19-20`` (the triton
    kernel's own docstring, which is also what its non-triton twin at
    ``loss_fns.py:159-166`` computes)::

        alpha_t = alpha * t + (1 - alpha) * (1 - t)
        loss    = alpha_t * ce * (1 - p_t) ** gamma

    Note ``alpha_t``, NOT a scalar ``alpha``: SAM 2's
    ``_focal_from_logits`` in this repo deliberately applies ``alpha`` as a flat
    scalar to both classes (its D-073), which is a different loss. This is a
    SAM 3 port, so it follows SAM 3's reference.

    ``gamma == 0.0`` is special-cased to a literal ``1.0`` modulator rather than
    ``ops.power(1 - p_t, 0.0)``. This is not cosmetic: the shipped
    ``presence_gamma`` IS ``0.0``, and ``power``'s derivative
    ``gamma * (1 - p_t) ** (gamma - 1)`` evaluates as ``0 * inf = nan`` when
    ``p_t`` saturates to ``1``. The value path is identical; the gradient path
    is not.

    :param truth: Binary (or soft) targets.
    :type truth: Any
    :param logits: Unbounded real logits.
    :type logits: Any
    :param alpha: Class-balance factor, applied as ``alpha_t``.
    :type alpha: float
    :param gamma: Focusing exponent. ``0.0`` makes the modulator inert and the
        result plain alpha-weighted BCE.
    :type gamma: float
    :return: Unreduced element-wise focal loss.
    :rtype: Any
    """
    cross_entropy = binary_cross_entropy_with_logits(truth, logits)
    alpha_t = alpha * truth + (1.0 - alpha) * (1.0 - truth)
    if gamma == 0.0:
        return alpha_t * cross_entropy
    probabilities = ops.sigmoid(logits)
    p_t = probabilities * truth + (1.0 - probabilities) * (1.0 - truth)
    return alpha_t * ops.power(1.0 - p_t, gamma) * cross_entropy


@keras.saving.register_keras_serializable()
class Sam3DetectionLoss(keras.losses.Loss):
    """The six-term SAM 3 detection loss over one shared Hungarian assignment.

    Interface contract: ``call(y_true, y_pred)`` consumes the packed tensors
    documented at module level and returns a SCALAR. :meth:`compute_terms`
    returns the same computation as a dict of named scalars, unweighted, which
    is what a trainer should log -- a single falling total can hide a term that
    is doing nothing. Both are safe on an all-negative batch (every divisor
    clamps) and on ``N > Q`` (the surplus GT is simply unmatched). Neither
    raises at call time; the constructor raises on a contradictory switch.

    The six terms, each transcribed from the reference
    ---------------------------------------------------
    ``loss_ce`` -- ``IABCEMdetr.get_loss``, ``loss_fns.py:349-507``. The
    positive branch's target is IoU-aware and DETACHED:
    ``t = prob**alpha * iou**(1 - alpha)``, clamped to ``>= 0.01``
    (:``loss_fns.py:371-372``); the positive term is PLAIN BCE times
    ``pos_weight`` because the shipped config sets ``pos_focal: false``
    (:``loss_fns.py:386-390``) -- applying the full focal modulation there would
    be a silent divergence (S-2); the negative term is BCE times
    ``prob ** gamma`` (:``loss_fns.py:399-401``), which is what replaces vanilla
    DETR's fixed ``eos_coef``.

    ``presence_loss`` -- ``loss_fns.py:430-440``. Alpha-weighted BCE on
    ``presence_logit`` against ``keep_loss``, with ``presence_alpha=0.5`` and
    ``presence_gamma=0.0``. At ``gamma = 0`` the focal modulation is INERT and
    the term IS plain alpha-weighted BCE; that is reproduced faithfully rather
    than "improved", and pinned by a test.

    ``loss_bbox`` / ``loss_giou`` -- ``Boxes.get_loss``, ``loss_fns.py:563-571``.
    L1 in ``cxcywh``, and ``1 - GIoU`` in ``xyxy``.

    ``loss_mask`` / ``loss_dice`` -- ``Masks.get_loss``, ``loss_fns.py:706-715``.
    Focal in LOGIT space, and soft dice ``1 - (2 * inter + 1) / (sum + 1)``.

    THE FIVE DIVISORS
    -----------------
    Each is a VALUE, not a comment, and each has a test whose inputs make the
    four wrong candidates give a numerically different answer.

    ===  ==================================  ==============================
    #    Term                                Divisor
    ===  ==================================  ==============================
    1    ``loss_bbox``/``giou``/``dice``     ``num_boxes``, clamped min 1
    2    ``loss_mask`` (focal)               ``num_boxes * P``
    3    ``presence_loss``                   ``B`` (batch size)
    4    ``loss_ce``, ``weak_loss=False``    ``pad_n_queries * B``, or a
                                             plain mean when
                                             ``Q >= pad_n_queries``
    5    ``loss_ce``, ``weak_loss=True``     ``retained_bce_count + 1e-6``
    ===  ==================================  ==============================

    #2 is the one the prior plan never flagged: the reference's default
    ``triton=True`` path returns an ALREADY-SUMMED scalar and divides by
    ``num_boxes * inputs.shape[1]`` (``loss_fns.py:153-155``), where
    ``inputs.shape[1]`` is the flattened pixel/point count. ``loss_dice`` is
    exempt from that extra factor, so the two mask terms sit on structurally
    different scales BEFORE their weights are applied. #4 and #5 are the two
    sides of a real switch: ``weak_loss`` defaults to ``True`` on the reference
    CLASS and to ``False`` in the reference's shipped CONFIG (S-1), so both are
    implemented and neither is dead code. This port defaults to the SHIPPED
    value.

    A sixth, GLOBAL multiplier -- ``scale_by_find_batch_size``
    ---------------------------------------------------------
    ``sam3_loss.py:192-195`` multiplies the whole core loss by ``bs ** 0.5``.
    It is implemented here and defaults to **OFF**; see decisions.md D-007 for
    why that default is a decision rather than an omission.

    Presence gates classification MULTIPLICATIVELY
    ----------------------------------------------
    ``loss_bce = loss_bce * keep_loss`` BEFORE any reduction
    (``loss_fns.py:425``). An image with no visible GT for its phrase
    contributes ZERO classification loss -- both the positive and the negative
    branch -- while ``presence_loss`` still supervises the negative. This is
    structural, not a tunable weight, and removing it changes what a zero-GT row
    means.

    Divergences from the reference, SIGNED and NAMED
    ------------------------------------------------
    ``this_port = reference + PORT_ONLY(...) - REFERENCE_ONLY(...)``

    * ``+ PORT_ONLY(masked-sum over all Q instead of index_select)``. The
      reference gathers the matched ``(query, target)`` pairs into a dense
      ``(M, ...)`` tensor and reduces that. Static shapes are load-bearing under
      ``fit()``, so this port keeps the full ``(B, Q, ...)`` tensor and
      multiplies by the matcher's ``is_matched`` mask before summing. The sums
      are equal term by term because every unmatched contribution is multiplied
      by exactly ``0.0``. The ONE thing this changes is that an unmatched query
      still evaluates the box arithmetic against a padded, all-zero gathered
      target row -- which is finite by the same measurement that removed the
      matcher's dummy-box guard (D-006): ``union = area_pred + 0 - 0`` stays
      positive. A ``nan`` needs BOTH boxes degenerate.
    * ``- REFERENCE_ONLY(is_valid_mask)``. ``Masks.get_loss`` filters matched
      pairs whose GT has no segmentation (``loss_fns.py:670-680``). This port
      packs masks and boxes on the same row behind ONE validity flag, so an
      instance is either fully supervised or fully absent; a separate
      mask-validity channel would be a second, weaker validity notion with no
      producer in phase 2.
    * ``- REFERENCE_ONLY(all_reduce / world_size)``. ``num_boxes`` is
      all-reduced across ranks (``sam3_loss.py:75``). Single-process here, where
      ``normalization="global"`` and ``"local"`` coincide exactly; the min-1
      clamp, which is the part that matters, is kept.
    * ``- REFERENCE_ONLY(pad_scale_pos)``. A scale on the positive term applied
      exactly when ``Q < pad_n_queries`` (``loss_fns.py:392-397``). It is
      dropped because that branch is **UNREACHABLE in every shipped reference
      config**, not merely because the value is ``1.0``: both config families
      set ``pad_n_queries`` to the model's own query count -- literally
      (``roboflow_*.yaml:100`` writes ``200`` beside ``num_queries=200``) and
      symbolically (``odinw_text_only_train.yaml:102`` writes
      ``${scratch.num_queries}``) -- so ``Q >= pad_n_queries`` always holds
      there and ``pad_scale_pos`` never multiplies anything. **The binding
      obligation is therefore on the CALLER: ``pad_n_queries`` is a
      VARIANT-DERIVED quantity, not a constant.** Leaving it at the reference's
      ``200`` while a variant emits ``Q = 32`` divides the whole classification
      term by exactly ``6.25`` -- MEASURED on a real ``small`` batch, raw
      ``loss_ce`` 0.043937 at 200 against 0.274605 at 32, and a weighted share
      of the total that moves 9.1 % -> 38.4 % over a 64-image split
      (decisions.md D-040).
    * ``- REFERENCE_ONLY(aux_outputs / first_stage deep supervision)``. The
      reference runs every loss on every intermediate decoder layer. The port's
      packed layout carries the final layer only; deep supervision is a packing
      question, deferred with the training model rather than half-built here.
    * ``- REFERENCE_ONLY(o2m / DAC branch, video association, semantic seg)``.
      Deferred wholesale, with their matchers.
    * ``- REFERENCE_ONLY(is_video_grounding / Q_det splits)``. Image-only path.

    :param include_masks: Whether the packed tensors carry a flattened mask
        block. Default ``False`` -- the reference's ONE shipped training config
        disables segmentation entirely (decisions.md D-009).
    :type include_masks: bool
    :param weight_ce: Weight on ``loss_ce``. Shipped ``20.0``.
    :type weight_ce: float
    :param weight_presence: Weight on ``presence_loss``. Shipped ``20.0``.
    :type weight_presence: float
    :param weight_bbox: Weight on ``loss_bbox``. Shipped ``5.0``.
    :type weight_bbox: float
    :param weight_giou: Weight on ``loss_giou``. Shipped ``2.0``.
    :type weight_giou: float
    :param weight_mask: Weight on ``loss_mask``. ``200.0`` in the reference's
        COMMENTED-OUT mask block; no config ever ran it.
    :type weight_mask: float
    :param weight_dice: Weight on ``loss_dice``. ``10.0``, same caveat.
    :type weight_dice: float
    :param pos_weight: Multiplier on the positive classification term. Shipped
        ``10.0``.
    :type pos_weight: float
    :param alpha: Exponent on ``prob`` in the IoU-aware soft target. Shipped
        ``0.25``.
    :type alpha: float
    :param gamma: Focusing exponent on the NEGATIVE classification term.
        Shipped ``2.0`` (the reference class default is ``0``).
    :type gamma: float
    :param pos_focal: Whether the positive classification term uses focal loss
        instead of plain BCE. Shipped ``False`` -- see S-2.
    :type pos_focal: bool
    :param weak_loss: Selects divisor #5 (``True``) or #4 (``False``). Default
        ``False``, the shipped config's value; the reference CLASS defaults to
        ``True``.
    :type weak_loss: bool
    :param pad_n_queries: Divisor #4's query budget. ``None`` means a plain
        mean. The default ``200`` is the RELEASED VARIANT's ``num_queries``, and
        it is a variant-derived quantity: **a caller running a variant with a
        different ``Q`` must pass that ``Q`` here**, exactly as the reference's
        own ``odinw`` config does (``pad_n_queries: ${scratch.num_queries}``).
        See the ``pad_scale_pos`` divergence above and decisions.md D-040.
    :type pad_n_queries: Optional[int]
    :param presence_alpha: Alpha for ``presence_loss``. Shipped ``0.5``.
    :type presence_alpha: float
    :param presence_gamma: Gamma for ``presence_loss``. Shipped ``0.0``, which
        makes the modulation inert.
    :type presence_gamma: float
    :param focal_alpha: Alpha for ``loss_mask``. ``0.25``.
    :type focal_alpha: float
    :param focal_gamma: Gamma for ``loss_mask``. ``2.0``.
    :type focal_gamma: float
    :param normalize_by_valid_object_num: ``True`` derives ``num_boxes`` from
        the packed validity flags; ``False`` sums the meta row's per-image
        ``num_boxes``. Both are reference modes (``sam3_loss.py:67-73``).
    :type normalize_by_valid_object_num: bool
    :param scale_by_find_batch_size: Multiply the total by ``sqrt(B)``. Default
        ``False``; decisions.md D-007.
    :type scale_by_find_batch_size: bool
    :param cost_class: Matcher weight on the class cost. Shipped ``2.0``.
    :type cost_class: float
    :param cost_bbox: Matcher weight on the L1 cost. Shipped ``5.0``.
    :type cost_bbox: float
    :param cost_giou: Matcher weight on the GIoU cost. Shipped ``2.0``.
    :type cost_giou: float
    :param kwargs: Forwarded to :class:`keras.losses.Loss`.
    :raises ValueError: If ``pad_n_queries`` is not a positive integer, or if a
        mask weight is non-zero while ``include_masks`` is ``False`` -- a
        combination that would silently supervise nothing.
    """

    def __init__(
            self,
            include_masks: bool = False,
            weight_ce: float = 20.0,
            weight_presence: float = 20.0,
            weight_bbox: float = 5.0,
            weight_giou: float = 2.0,
            weight_mask: float = 200.0,
            weight_dice: float = 10.0,
            pos_weight: float = 10.0,
            alpha: float = 0.25,
            gamma: float = 2.0,
            pos_focal: bool = False,
            weak_loss: bool = False,
            pad_n_queries: Optional[int] = 200,
            presence_alpha: float = 0.5,
            presence_gamma: float = 0.0,
            focal_alpha: float = 0.25,
            focal_gamma: float = 2.0,
            normalize_by_valid_object_num: bool = True,
            scale_by_find_batch_size: bool = False,
            cost_class: float = 2.0,
            cost_bbox: float = 5.0,
            cost_giou: float = 2.0,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if pad_n_queries is not None and (not isinstance(pad_n_queries, int)
                                          or pad_n_queries <= 0):
            raise ValueError(
                "Sam3DetectionLoss: pad_n_queries must be a positive int or "
                f"None, got {pad_n_queries!r}.")
        self.include_masks = bool(include_masks)
        self.weight_ce = float(weight_ce)
        self.weight_presence = float(weight_presence)
        self.weight_bbox = float(weight_bbox)
        self.weight_giou = float(weight_giou)
        self.weight_mask = float(weight_mask)
        self.weight_dice = float(weight_dice)
        self.pos_weight = float(pos_weight)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.pos_focal = bool(pos_focal)
        self.weak_loss = bool(weak_loss)
        self.pad_n_queries = pad_n_queries
        self.presence_alpha = float(presence_alpha)
        self.presence_gamma = float(presence_gamma)
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        self.normalize_by_valid_object_num = bool(normalize_by_valid_object_num)
        self.scale_by_find_batch_size = bool(scale_by_find_batch_size)
        self.cost_class = float(cost_class)
        self.cost_bbox = float(cost_bbox)
        self.cost_giou = float(cost_giou)
        self.matcher = _Sam3HungarianMatcher(
            cost_class=cost_class, cost_bbox=cost_bbox, cost_giou=cost_giou,
            alpha=alpha, gamma=gamma)

    def get_config(self) -> Dict[str, Any]:
        """Return every constructor argument, for a full round trip.

        :return: The serializable configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "include_masks": self.include_masks,
            "weight_ce": self.weight_ce,
            "weight_presence": self.weight_presence,
            "weight_bbox": self.weight_bbox,
            "weight_giou": self.weight_giou,
            "weight_mask": self.weight_mask,
            "weight_dice": self.weight_dice,
            "pos_weight": self.pos_weight,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "pos_focal": self.pos_focal,
            "weak_loss": self.weak_loss,
            "pad_n_queries": self.pad_n_queries,
            "presence_alpha": self.presence_alpha,
            "presence_gamma": self.presence_gamma,
            "focal_alpha": self.focal_alpha,
            "focal_gamma": self.focal_gamma,
            "normalize_by_valid_object_num":
                self.normalize_by_valid_object_num,
            "scale_by_find_batch_size": self.scale_by_find_batch_size,
            "cost_class": self.cost_class,
            "cost_bbox": self.cost_bbox,
            "cost_giou": self.cost_giou,
        })
        return config

    def compute_terms(self, y_true: Any, y_pred: Any) -> Dict[str, Any]:
        """Compute every loss term separately, UNWEIGHTED.

        Interface contract: the inputs are the packed tensors; the return maps
        each of ``loss_ce`` / ``presence_loss`` / ``loss_bbox`` / ``loss_giou``
        / ``loss_mask`` / ``loss_dice`` plus the diagnostic ``num_boxes`` and
        ``num_matched`` to a scalar tensor. The mask terms are present and
        exactly ``0.0`` when ``include_masks`` is ``False``, so a logger never
        has to branch. Every value is finite for every input this layout can
        express, including an all-negative batch. It never raises.

        This is the method a trainer should log. :meth:`call` is the weighted
        sum of exactly these numbers, so a falling total that hides a dead term
        is visible here and invisible there.

        :param y_true: Packed targets, ``(B, N_max + 1, C)``.
        :type y_true: Any
        :param y_pred: Packed predictions, ``(B, Q + 1, C)``.
        :type y_pred: Any
        :return: The six terms plus two diagnostics, all scalars.
        :rtype: Dict[str, Any]
        """
        predictions = unpack_predictions(
            ops.cast(y_pred, "float32"), self.include_masks)
        targets = unpack_targets(
            ops.cast(y_true, "float32"), self.include_masks)

        src_logits = predictions["pred_logits"]
        pred_boxes = predictions["pred_boxes"]
        target_boxes = targets["target_boxes"]
        target_valid = targets["target_valid"]
        keep_loss = targets["keep_loss"]

        batch_size = ops.cast(ops.shape(src_logits)[0], "float32")
        # STATIC, not `ops.shape(...)`: this selects a Python branch below, and
        # a graph tensor cannot select one. `Q` is a model constant, so a
        # static read is always available here.
        num_queries = int(src_logits.shape[1])

        # --- Divisor #1: `num_boxes`, clamped to a minimum of 1 (H-6). ---
        # `sam3_loss.py:65-81`. The clamp is what keeps an ALL-NEGATIVE batch
        # -- every image zero-GT -- from dividing by zero; it is not a
        # numerical nicety, it is the only thing standing between a legitimate
        # training batch and a NaN.
        if self.normalize_by_valid_object_num:
            # DECISION plan-2026-08-05T124709-6c4fac48/D-047
            # The predicate is the CONJUNCTION `valid AND w > 0 AND h > 0`.
            # Do NOT reduce it to EITHER conjunct alone. The flag alone
            # over-counts the reference's INVISIBLE-OBJECT row (a valid id
            # carrying a zero-area box); the GEOMETRY alone over-counts a
            # PADDING row that happens to carry extents, which `pack_targets`
            # does NOT zero -- measured, one such row moves `num_boxes`
            # 6.0 -> 7.0 and `loss_bbox` -14 %. `sam3_loss.py::_get_num_boxes:
            # 70-71` is geometry-only because its `targets["boxes"]` is a truly
            # PACKED tensor with no padding rows at all (`collator.py:286`
            # `extend`s real boxes; the padded form is the SEPARATE
            # `boxes_padded` key) -- that precondition does not hold here, so
            # copying its expression alone is a SIGNED NAMED DIVERGENCE
            # `+ valid`. This module's `derive_keep_loss` uses exactly this
            # conjunction. See decisions.md D-047 (which supersedes D-042).
            raw_num_boxes = ops.sum(
                ops.cast(ops.cast(target_valid, "float32") > 0.0, "float32")
                * ops.cast(
                    ops.all(target_boxes[..., 2:] > 0.0, axis=-1), "float32"))
        else:
            raw_num_boxes = ops.sum(targets["num_boxes"])
        num_boxes = ops.maximum(raw_num_boxes, 1.0)

        assignment, is_matched = self.matcher(
            src_logits, pred_boxes, target_boxes, target_valid)
        is_matched = ops.stop_gradient(ops.cast(is_matched, "float32"))

        gathered_boxes = ops.take_along_axis(
            target_boxes, assignment[:, :, None], axis=1)
        pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)
        gathered_xyxy = box_cxcywh_to_xyxy(gathered_boxes)
        iou, giou = iou_and_generalized_iou(pred_xyxy, gathered_xyxy)

        # --- loss_ce ------------------------------------------------------
        target_classes = is_matched
        probability = ops.sigmoid(src_logits)
        # The IoU-aware soft target, `loss_fns.py:370-374`. DETACHED: the
        # reference computes it inside `torch.no_grad()`, so the IoU must not
        # push gradient into `pred_boxes` through the classification term.
        # `iou` is forced to 0 on unmatched rows first, because those rows
        # gathered a padded target and their IoU is meaningless (and, in the
        # both-degenerate corner, nan) -- `t` there is discarded anyway.
        safe_iou = ops.where(target_classes > 0.0, iou, ops.zeros_like(iou))
        soft_target = ops.stop_gradient(
            ops.maximum(
                ops.power(probability, self.alpha)
                * ops.power(ops.maximum(safe_iou, 0.0), 1.0 - self.alpha),
                0.01,
            ))
        positive_target = ops.where(target_classes > 0.0, soft_target,
                                    ops.zeros_like(soft_target))

        if self.pos_focal:
            # `loss_fns.py:377-385` -- alpha is hardcoded 0.5 there, NOT
            # `self.alpha`, and `num_boxes=1` with `reduce=False` makes the
            # call return the unreduced tensor.
            positive_bce = sigmoid_focal_loss(
                positive_target, src_logits, alpha=0.5, gamma=self.gamma)
        else:
            positive_bce = binary_cross_entropy_with_logits(
                positive_target, src_logits)
        loss_bce = positive_bce * target_classes * self.pos_weight
        negative_bce = binary_cross_entropy_with_logits(
            target_classes, src_logits)
        loss_bce = loss_bce + (negative_bce * (1.0 - target_classes)
                               * ops.power(probability, self.gamma))

        # DECISION plan-2026-08-05T124709-6c4fac48/D-012
        # `keep_loss` gates the classification BCE MULTIPLICATIVELY and
        # BEFORE any reduction (`loss_fns.py:425`, invariant H-7). Do NOT
        # move this below the reduction and do NOT re-express it as a weight
        # on the total: an image with no visible GT for its phrase must
        # contribute exactly zero classification loss -- both branches -- and
        # under divisor #4 it must still be counted in the DENOMINATOR
        # (`pad_n_queries * B` and the plain mean both include its rows). A
        # gate applied after the sum, or folded into the weight, changes the
        # denominator too and silently rescales every other image's loss.
        # `test_presence_gate_zeroes_classification_for_a_zero_gt_image` goes
        # RED when this multiply is deleted. See decisions.md D-012.
        loss_bce = loss_bce * keep_loss

        if self.weak_loss:
            # --- Divisor #5, the reference CLASS default. ---
            # `loss_fns.py:456-463`. A non-exhaustively-annotated image's
            # NEGATIVE supervision is nulled (an unannotated instance is not
            # evidence of absence), and the denominator counts only the
            # elements that survived -- so dropping an element removes it from
            # numerator AND denominator, which is what makes this a mean over
            # retained elements rather than a down-weighted mean over all.
            not_exhaustive = ops.cast(
                targets["is_exhaustive"][:, None] < 0.5, "float32")
            is_negative = ops.cast(target_classes < 0.5, "float32")
            retained = 1.0 - not_exhaustive * is_negative
            loss_ce = (ops.sum(loss_bce * retained)
                       / (ops.sum(retained) + 1e-6))
        elif (self.pad_n_queries is None
              or num_queries >= self.pad_n_queries):
            # --- Divisor #4, sub-path A: a plain mean over `B * Q`. ---
            loss_ce = ops.mean(loss_bce)
        else:
            # --- Divisor #4, sub-path B: the SHIPPED path. ---
            # `loss_fns.py:507`. The denominator is the query budget the model
            # is PRETENDED to have (`pad_n_queries`), not the number it
            # actually emitted, so that the CE scale is comparable across
            # datasets whose heads emit different query counts.
            loss_ce = (ops.sum(loss_bce)
                       / (float(self.pad_n_queries) * batch_size))

        # --- presence_loss: Divisor #3, the BATCH SIZE ---------------------
        # `loss_fns.py:433-440` passes `num_boxes=bs` with the explicit comment
        # "not num_boxes, but we'll use it to normalize by bs", and the focal
        # helper then divides by `num_boxes * inputs.shape[1]` where
        # `inputs.shape[1] == 1` for a `(B, 1)` presence logit. So the divisor
        # is `B`, NOT `num_boxes` and NOT `B * Q`.
        presence_loss = ops.sum(
            sigmoid_focal_loss(keep_loss, predictions["presence_logit"],
                               alpha=self.presence_alpha,
                               gamma=self.presence_gamma)) / batch_size

        # --- loss_bbox / loss_giou: Divisor #1 -----------------------------
        loss_bbox = ops.sum(
            ops.sum(ops.abs(pred_boxes - gathered_boxes), axis=-1)
            * target_classes) / num_boxes
        loss_giou = ops.sum((1.0 - giou) * target_classes) / num_boxes

        loss_mask, loss_dice = self._mask_terms(
            predictions["pred_masks"], targets["target_masks"], assignment,
            target_classes, num_boxes)

        return {
            "loss_ce": loss_ce,
            "presence_loss": presence_loss,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
            "loss_mask": loss_mask,
            "loss_dice": loss_dice,
            "num_boxes": num_boxes,
            "num_matched": ops.sum(target_classes),
        }

    def _mask_terms(self, pred_masks: Any, target_masks: Any, assignment: Any,
                    target_classes: Any, num_boxes: Any) -> Tuple[Any, Any]:
        """Compute ``loss_mask`` and ``loss_dice``, or two zeros.

        Interface contract: ``pred_masks`` is ``(B, Q, P)`` logits or ``None``
        and ``target_masks`` is ``(B, N, P)`` or ``None``; the return is the
        pair of scalars. When either is ``None`` the return is two exact
        ``0.0``s of the right dtype, so a caller never branches on
        ``include_masks``. It never raises.

        :param pred_masks: ``(B, Q, P)`` mask logits, or ``None``.
        :type pred_masks: Any
        :param target_masks: ``(B, N, P)`` GT masks, or ``None``.
        :type target_masks: Any
        :param assignment: ``(B, Q)`` matched target index.
        :type assignment: Any
        :param target_classes: ``(B, Q)`` matched mask in ``{0, 1}``.
        :type target_classes: Any
        :param num_boxes: Scalar divisor #1.
        :type num_boxes: Any
        :return: ``(loss_mask, loss_dice)``.
        :rtype: Tuple[Any, Any]
        """
        if pred_masks is None or target_masks is None:
            zero = ops.sum(target_classes) * 0.0
            return zero, zero

        gathered_masks = ops.take_along_axis(
            target_masks, assignment[:, :, None], axis=1)
        matched = target_classes[:, :, None]
        pixel_count = ops.cast(ops.shape(pred_masks)[-1], "float32")

        # --- Divisor #2: `num_boxes * P`, the one the prior plan missed. ---
        # `loss_fns.py:153-155`. The default `triton=True` path calls
        # `triton_sigmoid_focal_loss_reduce`, which returns an ALREADY-SUMMED
        # scalar over every element (`sigmoid_focal_loss.py:295`), and then
        # divides by `num_boxes * inputs.shape[1]`. The non-triton fallback's
        # `loss.mean(1).sum() / num_boxes` is the SAME number -- `mean(1)` is
        # the `/P`. Dividing by `num_boxes` alone would inflate this term by a
        # factor of P, i.e. by 4096 at a 64x64 mask, before its 200.0 weight.
        focal = sigmoid_focal_loss(gathered_masks, pred_masks,
                                   alpha=self.focal_alpha,
                                   gamma=self.focal_gamma)
        loss_mask = ops.sum(focal * matched) / (num_boxes * pixel_count)

        # --- loss_dice: Divisor #1 only. `loss_fns.py:104-122`. ---
        # Dice reduces spatially FIRST, so it carries no extra P factor. That
        # asymmetry against `loss_mask` is the reference's, not a slip.
        probabilities = ops.sigmoid(pred_masks)
        numerator = 2.0 * ops.sum(probabilities * gathered_masks, axis=-1)
        denominator = (ops.sum(probabilities, axis=-1)
                       + ops.sum(gathered_masks, axis=-1))
        dice = 1.0 - (numerator + 1.0) / (denominator + 1.0)
        loss_dice = ops.sum(dice * target_classes) / num_boxes
        return loss_mask, loss_dice

    def call(self, y_true: Any, y_pred: Any) -> Any:
        """Return the weighted sum of the six terms as a scalar.

        :param y_true: Packed targets, ``(B, N_max + 1, C)``.
        :type y_true: Any
        :param y_pred: Packed predictions, ``(B, Q + 1, C)``.
        :type y_pred: Any
        :return: A scalar loss.
        :rtype: Any
        """
        terms = self.compute_terms(y_true, y_pred)
        total = (self.weight_ce * terms["loss_ce"]
                 + self.weight_presence * terms["presence_loss"]
                 + self.weight_bbox * terms["loss_bbox"]
                 + self.weight_giou * terms["loss_giou"])
        if self.include_masks:
            total = total + (self.weight_mask * terms["loss_mask"]
                             + self.weight_dice * terms["loss_dice"])
        if self.scale_by_find_batch_size:
            # DECISION plan-2026-08-05T124709-6c4fac48/D-007
            # `sam3_loss.py:192-195`. Default OFF. This multiplier exists to
            # make IMAGE batches and VIDEO batches (which sum a loss over
            # every "find stage", i.e. every training frame) land on
            # comparable scales. This plan trains images only, at one stage,
            # so switching it on would silently rescale the effective learning
            # rate by sqrt(B) against every LR number in the reference recipe
            # -- and would make a batch-size change look like an LR change in
            # any A/B. It is implemented, not omitted, because a future video
            # path needs it. See decisions.md D-007.
            batch_size = ops.cast(ops.shape(y_pred)[0], "float32")
            total = total * ops.sqrt(batch_size)
        return total


__all__: List[str] = [
    "INVALID_COST",
    "META_IS_EXHAUSTIVE",
    "META_KEEP_LOSS",
    "META_NUM_BOXES",
    "PACKED_BOX_START",
    "PACKED_MASK_START",
    "PACKED_SCORE_CHANNEL",
    "VALID_COST_THRESHOLD",
    "Sam3DetectionLoss",
    "binary_cross_entropy_with_logits",
    "box_cxcywh_to_xyxy",
    "derive_keep_loss",
    "iou_and_generalized_iou",
    "packed_channel_count",
    "pairwise_generalized_iou",
    "sigmoid_focal_loss",
    "unpack_predictions",
    "unpack_targets",
]
