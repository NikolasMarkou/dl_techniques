"""
DocScanner Progressive-Rectification Sequence Loss
==================================================

The training objective of DocScanner's rectification module, Eq. 9-14 of
arXiv:2110.14968v2 (Feng et al., *DocScanner: Robust Document Image
Rectification with Progressive Learning*)::

    L        = sum_{k=1..K} gamma^(K-k) * L^(k)                     (Eq. 9)
    L^(k)    = L_f^(k) + alpha * L_line^(k)                         (Eq. 10)
    L_f^(k)  = || f_gt - f^k ||_1                                   (Eq. 11)
    L_line   = (1/H) sum_i L_row(i) + (1/W) sum_j L_col(j)          (Eq. 14)

with ``K = 12``, ``gamma = 0.85`` and ``alpha = 0.5``.

**There is no upstream implementation of any of this.** The released
DocScanner repository (https://github.com/fh2019ustc/DocScanner) ships
inference only: no training loop, no loss, no optimizer. Every number and every
reading below is taken from the paper's own text, and the two places where the
paper is ambiguous are named as such rather than papered over (see
*Two readings this file had to choose* at the bottom of this docstring).

The weight direction
--------------------
``gamma^(K-k)`` puts weight **exactly 1.0 on the LAST iteration** and
``gamma^(K-1)`` on the first. That is the paper's own wording -- *"where
gamma^(K-k) is the weight of the k-th iteration which increases exponentially
(gamma < 1)"* -- and the RAFT convention DocScanner inherits. Inverting it to
``gamma^(k-1)`` trains the network to be good at its first guess and to stop
caring about its final answer, produces a perfectly finite decreasing loss
curve, and is invisible to every shape, dtype, finiteness and serialization
check. :class:`TestTheLastIterationCarriesWeightOne` in
``tests/test_losses/test_doc_scanner_flow_sequence_loss.py`` exists for exactly
that mutation.

The circle-consistency (line) term
----------------------------------
Quoting the paper: *"we first map the pixels of i-th row (i.e. line_s) in
ground truth document image to I_D, based on the predicted backward warping
flow f^k. Secondly, we map these pixels back to the ground truth document image
again, using the ground truth forward warping flow g. After the above two
steps, we get a curved line line_c, which shall be the straight line line_s
when the backward warping flow in the first step is perfectly estimated."*

So the term is a **two-step composed warp**, and the composition is::

    c = g( f^k )        i.e. sample_at_pixel_coords(g, f_k)

``f^k`` is the network's predicted BACKWARD map: read at flat-document pixel
``(x, y)`` it gives the distorted-image coordinate that pixel came from. Since
the straight line ``line_s`` is a row of the flat document, its points sit on
integer pixels and step one is a plain read of ``f^k`` -- no resampling. Step
two evaluates the ground-truth FORWARD map ``g`` at those (generally
fractional) distorted coordinates, which is one bilinear sample. Under a
perfect ``f^k`` the round trip is the identity, ``c`` is the identity
coordinate grid, and every row of it has a constant y-coordinate -- i.e. zero
variance. The loss is that variance:

* ``L_row(i)`` = variance of the **y** component of ``c`` along row ``i``;
* ``L_col(j)`` = variance of the **x** component of ``c`` along column ``j``;
* ``L_line``   = mean over rows of ``L_row`` + mean over columns of ``L_col``.

Units, and what that means for ``alpha``
----------------------------------------
Everything here is in **absolute pixels**, because that is what
:class:`~dl_techniques.models.vision.image_restoration.doc_scanner.model.DocScannerRectifier`
emits and what a Doc3D/UVDoc-style backward map is stored as. ``L_f`` is
therefore a mean pixel displacement and ``L_line`` a mean squared pixel
deviation, so the two terms do not share units and ``alpha = 0.5`` is not a
dimensionless blend. That is the paper's specification and it is transcribed,
not "fixed"; a port that normalizes the coordinates to ``[-1, 1]`` first is a
different objective with a different effective ``alpha``.

Wiring: NO custom ``train_step``
--------------------------------
The iteration sequence is a model OUTPUT, not something this loss reconstructs:
``DocScannerRectifier(x, training=True)`` returns ``(B, K, H, W, 2)``. So the
whole objective is reachable through stock ``compile(loss=...)``/``fit()`` and
this port adds no custom ``train_step`` anywhere::

    from dl_techniques.losses import DocScannerFlowSequenceLoss
    from dl_techniques.models.vision.image_restoration.doc_scanner import (
        DocScannerRectifier,
    )

    model = DocScannerRectifier.from_variant("docscanner-l")
    model.compile(optimizer="adam", loss=DocScannerFlowSequenceLoss())
    # y is the 4-channel stack [f_gt(2), g(2)], y_pred is (B, 12, H, W, 2)
    model.fit(x, y, epochs=1)

Note that ``y_pred`` only has the sequence axis under ``training=True``. A
``predict()``/``evaluate()`` call returns the last iteration alone,
``(B, H, W, 2)``, and this loss raises a ValueError naming that situation
rather than silently broadcasting.

Two readings this file had to choose
------------------------------------
1. **``|| . ||_1`` is implemented as a MEAN, not a raw sum.** Eq. 11 writes an
   L1 norm. At 288x288x2 a raw sum is ~166k terms, which would make
   ``alpha = 0.5`` meaningless by six orders of magnitude and make the loss
   scale with resolution. RAFT's ``sequence_loss`` -- the direct ancestor of
   Eq. 9 -- means it, and Eq. 14's own ``1/H``/``1/W`` factors show the paper
   is averaging on the other term too. Documented here as a reading; see
   ``decisions.md`` D-033 of the porting plan.
2. **Eq. 14's row sum runs over ALL H rows here.** The ar5iv extraction of
   Eq. 14 literally shows the row index running ``i = 1..W`` rather than
   ``1..H`` (finding S-5 of the porting plan). At the paper's own square
   288x288 setting ``H == W``, so the ambiguity is inert there and cannot be
   resolved from the text. This implementation averages over every row and
   every column, i.e. reads it as ``i = 1..H``, because the literal ``1..W``
   reading is out of range whenever ``W > H`` and regularizes only part of the
   image whenever ``W < H``. The ambiguity is real and is NOT silently
   corrected: it is recorded here, in the class docstring, and in D-034.

References:
    - Feng et al., 2021. DocScanner: Robust Document Image Rectification with
      Progressive Learning. (https://arxiv.org/abs/2110.14968), v2, Eq. 9-14.
    - Teed & Deng, 2020. RAFT: Recurrent All-Pairs Field Transforms for
      Optical Flow. (https://arxiv.org/abs/2003.12039) -- the sequence-loss
      convention Eq. 9 follows.
"""

from typing import Any, Dict, List

import keras

from dl_techniques.utils.keras_registration import register_dl_technique

# DECISION plan-2026-09-10T065432-05fcb6dd/D-031: this loss lives in `losses/`
# (the repo's home for every loss) and reaches ACROSS into `models/` for the
# sampler and the constants. That direction is new -- no other module in
# `losses/` imports from `models/` -- and it is deliberate:
#
#   * The alternative, a `doc_scanner/losses.py` inside the model package,
#     would put the one loss of this port in the one place a user of this repo
#     does not look for a loss, and `losses/CLAUDE.md` states the convention
#     explicitly ("All losses are exported from `__init__.py`").
#   * Re-deriving the bilinear sampler here instead is the DRY violation this
#     port has already refused twice: `warp.py` exists precisely so that the
#     align_corners=True pixel-coordinate convention (D-009, F-14) has exactly
#     one definition. A second sampler would be a second convention.
#
# MEASURED before adopting it: `dl_techniques.losses` does not appear in
# `sys.modules` after importing the doc_scanner package, so this is NOT a
# cycle, and the marginal import cost is ~3 ms (both packages pull `keras` and
# `dl_techniques.layers`, which dominate at ~3.2 s either way).
#
# Do NOT let `doc_scanner/` import `dl_techniques.losses` at module scope: THAT
# would close the cycle this import currently only half-forms. If the trainers
# of step 15 need the loss, they must import it from `dl_techniques.losses`
# themselves, which they do.
from dl_techniques.models.vision.image_restoration.doc_scanner.components import (
    FLOW_CHANNELS,
    LINE_LOSS_WEIGHT,
    REFINE_ITERATIONS,
    SEQUENCE_LOSS_GAMMA,
)
from dl_techniques.models.vision.image_restoration.doc_scanner.warp import (
    sample_at_pixel_coords,
)

# ---------------------------------------------------------------------

#: Channel count of the ``y_true`` stack: ``[f_gt(2), g(2)]``.
_TARGET_CHANNELS: int = 2 * FLOW_CHANNELS

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.losses.doc_scanner_flow_sequence_loss")
class DocScannerFlowSequenceLoss(keras.losses.Loss):
    """Iteration-weighted flow + circle-consistency loss for DocScanner.

    Implements Eq. 9-14 of arXiv:2110.14968v2 over the whole refinement
    sequence a :class:`DocScannerRectifier` emits under ``training=True``. See
    the module docstring for the derivation, the units and the two readings
    this implementation had to choose (the L1-as-mean reading, and Eq. 14's
    ``i = 1..W`` row-index ambiguity, finding S-5 -- neither is silently
    corrected).

    The three constants default to the paper's values, which live in
    ``doc_scanner/components.py`` next to every other cited constant of the
    port rather than being re-typed here.

    Args:
        iters: ``K``, the number of refinement iterations the sequence carries.
            Must be a positive int and must match ``y_pred.shape[1]``.
            Defaults to
            :data:`~dl_techniques.models.vision.image_restoration.doc_scanner.components.REFINE_ITERATIONS`
            (12).
        gamma: The paper's ``lambda``, the exponential decay applied BACKWARDS
            from the last iteration: iteration ``k`` (1-based) is weighted
            ``gamma ** (K - k)``, so the last carries exactly ``1.0``. Must be
            in ``(0, 1]``. Defaults to
            :data:`~dl_techniques.models.vision.image_restoration.doc_scanner.components.SEQUENCE_LOSS_GAMMA`
            (0.85).
        line_weight: The paper's ``alpha``, the weight of the
            circle-consistency term relative to the L1 flow term. Must be
            non-negative; ``0.0`` disables the term (and, per the plan's
            pre-mortem, is the honest setting if the term cannot be shown to
            bite on real data). Defaults to
            :data:`~dl_techniques.models.vision.image_restoration.doc_scanner.components.LINE_LOSS_WEIGHT`
            (0.5).
        name: Loss name.
        **kwargs: Forwarded to :class:`keras.losses.Loss` (``reduction``,
            ``dtype``).

    Input shapes:
        - ``y_true``: ``(B, H, W, 4)`` -- the channel-wise stack
          ``[f_gt_x, f_gt_y, g_x, g_y]``. ``f_gt`` is the ground-truth BACKWARD
          map and ``g`` the ground-truth FORWARD map (Eq. 12), both in absolute
          pixel units, both in ``(x, y)`` channel order -- the order
          :func:`~dl_techniques.models.vision.image_restoration.doc_scanner.warp.coords_grid`
          emits and the only order the sampler accepts.
        - ``y_pred``: ``(B, K, H, W, 2)`` -- the refinement sequence, OLDEST
          FIRST, in absolute pixel units. This is exactly what
          ``DocScannerRectifier(x, training=True)`` returns.

    Output shape:
        ``(B,)`` before ``reduction``; a scalar under the default
        ``"sum_over_batch_size"``.

    Raises:
        ValueError: On construction, for a non-positive ``iters``, a ``gamma``
            outside ``(0, 1]`` or a negative ``line_weight``. On call, when
            ``y_pred`` is not rank 5 (the common cause is a ``training=False``
            forward, which emits the last iteration only), when its sequence
            axis is not ``iters``, or when ``y_true`` does not carry exactly 4
            channels.

    Example:
        >>> import numpy as np
        >>> from dl_techniques.losses import DocScannerFlowSequenceLoss
        >>> loss = DocScannerFlowSequenceLoss(iters=3)
        >>> y_true = np.zeros((2, 8, 8, 4), dtype="float32")
        >>> y_pred = np.zeros((2, 3, 8, 8, 2), dtype="float32")
        >>> float(loss(y_true, y_pred))  # doctest: +SKIP
        0.0
    """

    def __init__(
            self,
            iters: int = REFINE_ITERATIONS,
            gamma: float = SEQUENCE_LOSS_GAMMA,
            line_weight: float = LINE_LOSS_WEIGHT,
            name: str = "doc_scanner_flow_sequence_loss",
            **kwargs: Any
    ) -> None:
        super().__init__(name=name, **kwargs)

        if int(iters) <= 0:
            raise ValueError(
                f"`iters` must be a positive integer, got {iters}."
            )
        if not 0.0 < float(gamma) <= 1.0:
            raise ValueError(
                "`gamma` is an exponential decay applied backwards from the "
                f"last iteration and must lie in (0, 1]; got {gamma}."
            )
        if float(line_weight) < 0.0:
            raise ValueError(
                "`line_weight` (the paper's alpha) must be non-negative; got "
                f"{line_weight}."
            )

        self.iters = int(iters)
        self.gamma = float(gamma)
        self.line_weight = float(line_weight)

    # -----------------------------------------------------------------

    def _iteration_weights(self) -> List[float]:
        """Return the ``K`` per-iteration weights, OLDEST FIRST.

        Returns:
            A list of ``K`` Python floats, ``[gamma**(K-1), ..., gamma**1,
            1.0]``. Element ``-1`` is exactly ``1.0`` by construction, for
            every ``gamma``.
        """
        # DECISION plan-2026-09-10T065432-05fcb6dd/D-032: the exponent is
        # `K - 1 - index` on a 0-BASED index, i.e. `gamma ** (K - k)` on the
        # paper's 1-based `k`. The LAST element is `gamma ** 0 == 1.0` and the
        # FIRST is `gamma ** (K - 1)`.
        #
        # Do NOT write `gamma ** index`. That inversion is the single most
        # likely defect in this file: it is one character away, it keeps the
        # loss finite and decreasing, it keeps every shape, it serializes, and
        # it trains a progressive-refinement network to optimize its FIRST
        # guess and to stop caring about the answer it actually returns at
        # inference. The paper's own words are that the weight "increases
        # exponentially" toward the final iteration (Eq. 9), and RAFT's
        # `sequence_loss` -- which Eq. 9 follows -- is the same direction.
        #
        # Guarded by `TestTheLastIterationCarriesWeightOne`, which pins the
        # weight vector element-wise AND asserts, through the loss itself, that
        # a single wrong iteration costs strictly MORE the later it sits in the
        # sequence. See decisions.md D-032 for the RED proof.
        return [
            self.gamma ** (self.iters - 1 - index)
            for index in range(self.iters)
        ]

    def _line_loss(
            self,
            composed: "keras.KerasTensor"
    ) -> "keras.KerasTensor":
        """Circle-consistency term of Eq. 13/14 on an already-composed field.

        Args:
            composed: ``(N, H, W, 2)`` result of the two-step warp, in ``(x,
            y)`` channel order. Under a perfect prediction this is the identity
            coordinate grid.

        Returns:
            ``(N,)``. Zero exactly when every row of ``composed`` has a
            constant y-coordinate and every column a constant x-coordinate,
            i.e. when the round-tripped lines are straight.
        """
        # DECISION plan-2026-09-10T065432-05fcb6dd/D-034: WHICH COMPONENT goes
        # with WHICH AXIS, and both of them are inert-looking when swapped.
        #
        # A row of the flat page is straight iff its Y coordinate is constant
        # ALONG the row (axis 2, the W axis). Taking the variance of the X
        # component along a row instead measures how evenly the round trip
        # spread the points out along the line, which is a resampling statistic,
        # not a curvature one, and is nonzero for a perfectly straight but
        # non-uniformly sampled line. Symmetrically for columns.
        #
        # Do NOT "simplify" this to a single `var(composed, axis=(1, 2))` over
        # both components at once either: that is a different quantity (it adds
        # the two cross terms and drops the per-row/per-column structure) and it
        # is ALSO zero on the identity-grid fixture, so the analytic
        # zero-loss test cannot tell the two apart. Only the curved-line arm
        # can. Guarded by `TestTheLineTermPenalizesCurvature`.
        #
        # S-5: the mean below runs over ALL H rows and ALL W columns. Eq. 14's
        # ar5iv extraction literally indexes the row sum `i = 1..W`; at the
        # paper's square 288x288 that is the same statement, and off-square it
        # is out of range for W > H. See the module docstring and D-034.
        row_variance = keras.ops.var(composed[..., 1], axis=2)
        column_variance = keras.ops.var(composed[..., 0], axis=1)
        return (
            keras.ops.mean(row_variance, axis=1)
            + keras.ops.mean(column_variance, axis=1)
        )

    # -----------------------------------------------------------------

    def call(
            self,
            y_true: "keras.KerasTensor",
            y_pred: "keras.KerasTensor"
    ) -> "keras.KerasTensor":
        """Compute the per-sample sequence loss.

        Args:
            y_true: ``(B, H, W, 4)``, the ``[f_gt(2), g(2)]`` stack.
            y_pred: ``(B, K, H, W, 2)``, the refinement sequence, oldest first.

        Returns:
            ``(B,)`` per-sample loss.

        Raises:
            ValueError: If the static ranks/extents do not match the contract
                documented on the class.
        """
        pred_shape = y_pred.shape
        if len(pred_shape) != 5:
            raise ValueError(
                "`y_pred` must be the whole refinement sequence, "
                "(batch, iters, height, width, 2). Got rank "
                f"{len(pred_shape)} with shape {tuple(pred_shape)}. The usual "
                "cause is a forward pass with `training=False`, which returns "
                "the LAST iteration only -- this loss is a training objective "
                "and needs `DocScannerRectifier(x, training=True)`."
            )
        if pred_shape[1] is not None and int(pred_shape[1]) != self.iters:
            raise ValueError(
                f"This loss was configured for iters={self.iters}, but "
                f"`y_pred` carries {int(pred_shape[1])} iterations. The two "
                "must agree: the per-iteration weights are positional and a "
                "mismatch would silently reweight the sequence."
            )
        true_shape = y_true.shape
        if len(true_shape) != 4 or (
                true_shape[-1] is not None
                and int(true_shape[-1]) != _TARGET_CHANNELS
        ):
            raise ValueError(
                "`y_true` must be the 4-channel stack [f_gt_x, f_gt_y, g_x, "
                f"g_y] of shape (batch, height, width, {_TARGET_CHANNELS}); "
                f"got {tuple(true_shape)}. The forward map `g` is the second "
                "half and is required by the circle-consistency term (Eq. 12)."
            )

        flow_gt = y_true[..., :FLOW_CHANNELS]
        forward_gt = y_true[..., FLOW_CHANNELS:]

        # Fold the sequence axis into the batch so the bilinear sampler and
        # both reductions run ONCE over `B * K` fields instead of K times over
        # B. `reshape` is row-major, so entry `b * K + k` of the folded axis is
        # `(b, k)`, and `repeat(..., K, axis=0)` replicates `g` in exactly that
        # order -- `tile` would interleave it the other way and silently pair
        # every prediction with the wrong sample's forward map.
        spatial = keras.ops.shape(y_pred)
        batch, height, width = spatial[0], spatial[2], spatial[3]
        folded_pred = keras.ops.reshape(
            y_pred, (batch * self.iters, height, width, FLOW_CHANNELS)
        )
        folded_flow_gt = keras.ops.repeat(flow_gt, self.iters, axis=0)
        folded_forward_gt = keras.ops.repeat(forward_gt, self.iters, axis=0)

        # DECISION plan-2026-09-10T065432-05fcb6dd/D-033: Eq. 11 writes an L1
        # NORM, and this is a MEAN, deliberately. Do NOT "restore the paper's
        # sum": at 288x288x2 a sum is ~166k terms, which puts this term six
        # orders of magnitude above the `L_line` it is blended with at
        # `alpha = 0.5` and makes the whole objective scale with the crop size.
        # Eq. 14 carries explicit 1/H and 1/W factors, so the paper is
        # averaging on the other term, and RAFT's `sequence_loss` -- the
        # ancestor of Eq. 9 -- means its L1 too. The reading, and its
        # consequence (that `alpha` is not dimensionless), are stated in the
        # module docstring under "Two readings this file had to choose".
        # See decisions.md D-033.
        flow_term = keras.ops.mean(
            keras.ops.abs(folded_flow_gt - folded_pred), axis=(1, 2, 3)
        )

        if self.line_weight > 0.0:
            # DECISION plan-2026-09-10T065432-05fcb6dd/D-034: the composition is
            # `g(f^k)` -- the GT FORWARD map sampled AT the predicted BACKWARD
            # map -- and not `f^k(g)`. Eq. 13's two steps are ordered: the
            # straight line lives in the FLAT page, `f^k` carries it into the
            # distorted image (a plain read, because the line's points are
            # integer pixels of the flat page), and only then does `g` carry it
            # back. Swapping the two arguments is shape-identical, finite,
            # differentiable and still exactly zero on the identity fixture, so
            # nothing but a genuinely asymmetric fixture can see it.
            composed = sample_at_pixel_coords(
                folded_forward_gt, folded_pred
            )
            line_term = self._line_loss(composed)
            per_iteration = flow_term + self.line_weight * line_term
        else:
            per_iteration = flow_term

        per_iteration = keras.ops.reshape(
            per_iteration, (batch, self.iters)
        )
        weights = keras.ops.convert_to_tensor(
            self._iteration_weights(), dtype=per_iteration.dtype
        )
        return keras.ops.sum(per_iteration * weights, axis=1)

    # -----------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Return every constructor argument needed to recreate this loss.

        Returns:
            Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            "iters": self.iters,
            "gamma": self.gamma,
            "line_weight": self.line_weight,
        })
        return config
