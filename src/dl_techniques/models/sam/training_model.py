"""
`SAMTrainingModel` -- the trainable wrapper around :class:`SAM`.

Why a wrapper exists at all
---------------------------
``SAM.call`` **cannot be traced**. ``postprocess_masks`` runs unconditionally at
the end of ``SAM.call``, and its ``ops.image.resize`` raises
``TypeError: len is not well defined for a symbolic Tensor`` under ``fit()``'s
graph mode. Reading only the ``low_res_logits`` key does *not* avoid it -- the
postprocess runs regardless of which key the caller consumes. A Python-tuple
``original_size`` cannot be passed either, because ``Layer.__call__`` rejects
non-tensor positional arguments.

So the training path calls SAM's submodules directly:

    ``preprocess -> image_encoder -> prompt_encoder -> mask_decoder``

and returns the two differentiable tensors as a dict, which stock
``compile()``/``fit()`` trains provided ``loss=`` is itself a **dict keyed to
the output names** with a matching dict ``y_true``. (The often-repeated claim
that a dict ``y_pred`` cannot be trained by stock ``fit()`` is over-general: it
fails for exactly one configuration, a single ``Loss`` object plus a bare-tensor
``y_true``.)

Usage
-----
    >>> trainer = SAMTrainingModel(sam, multimask_output=False)
    >>> trainer.compile(
    ...     optimizer="adam",
    ...     loss={"low_res_logits": SAMMaskLoss(), "iou_supervision": SAMIoULoss()},
    ...     loss_weights={"low_res_logits": 1.0, "iou_supervision": 1.0},
    ... )
    >>> trainer.fit(dataset)          # dataset yields (inputs_dict, y_true_dict)

    ``loss=`` may key a SUBSET of the output keys (measured), which is why
    ``iou_predictions`` can stay unsupervised while ``iou_supervision`` carries
    the IoU term. ``SAMMaskLoss`` already carries the focal:dice mix internally,
    so ``loss_weights`` only balances mask against IoU.

This module makes **no accuracy claim**. It proves that the training path runs
with live gradients; it does not claim SAM trained to any quality.
"""

from typing import Any, Dict, Optional, Tuple

import keras
from keras import ops

from ...losses.sam_mask_loss import match_mask_axis
from .model import SAM

# ---------------------------------------------------------------------------
# Public output-key constants. Shared with the losses and the trainer so the
# `compile(loss={...})` dict cannot drift from what `call` actually returns.
# ---------------------------------------------------------------------------
#: Key of the low-resolution mask logits -- **the** training target.
LOW_RES_LOGITS = "low_res_logits"
#: Key of the predicted IoU scores.
IOU_PREDICTIONS = "iou_predictions"
#: Key of the packed IoU supervision pair, emitted ONLY when `gt_mask` is in
#: the inputs. Shape `(B, M, 2)`: `[..., 0]` is the predicted IoU (the same
#: values as `iou_predictions`, differentiable) and `[..., 1]` is the achieved
#: IoU, already `stop_gradient`-ed. See `SAMIoULoss` and D-036.
IOU_SUPERVISION = "iou_supervision"
#: Keys `call` ALWAYS returns, in a stable order.
OUTPUT_KEYS: Tuple[str, ...] = (LOW_RES_LOGITS, IOU_PREDICTIONS)
#: Keys `call` returns conditionally on the inputs it was given.
OPTIONAL_OUTPUT_KEYS: Tuple[str, ...] = (IOU_SUPERVISION,)

#: Input keys `call` understands. `image` is required; the prompt keys are
#: optional but at least one prompt must be present.
INPUT_IMAGE = "image"
INPUT_POINT_COORDS = "point_coords"
INPUT_POINT_LABELS = "point_labels"
INPUT_BOXES = "boxes"
#: Optional binary GT mask stack `(B, M, h, w)` at `low_res_logits` resolution.
#: Supplying it turns the `iou_supervision` output on.
INPUT_GT_MASK = "gt_mask"

#: Logit threshold at which a mask pixel counts as foreground when the achieved
#: IoU is measured. Matches `SAM.mask_threshold`'s own default.
IOU_MASK_THRESHOLD = 0.0
#: Added to the IoU numerator and denominator so an empty union gives 1.0.
IOU_SMOOTH = 1e-6

#: Rounds the CLASS defaults to. **1 = refinement off.** Refinement is opt-in,
#: not opt-out, and that is a measured decision rather than timidity (D-037):
#: at any value > 1 the output's mask axis is `M * rounds`, the wrapper stops
#: being value-exactly equivalent to an eager `SAM.call` (the A-2 guard), and
#: the `.keras` round-trip stops being value-exact because the sampling
#: advances a random state. Those three properties are the wrapper's inference-
#: shaped contract; a caller who wants refinement is training, and says so.
DEFAULT_REFINEMENT_ROUNDS = 1
#: Rounds the TRAINER ships. Derived here by measurement, not copied -- the
#: paper's "11 rounds" could not be verified from any primary source (F-5 item
#: 12), and 11 rounds is 11 decoder passes per step. Measured on the reduced
#: fixture, one `fit()` step each:
#:
#:   rounds=1  170/201 moved   rounds=2  181/201   rounds=3  181/201
#:                                                 rounds=4  181/201
#:
#: **Variable coverage saturates at 2**, not 3: the whole `mask_downscaling`
#: stack (10 vars) and the background-point type embedding become reachable the
#: moment ONE round feeds a mask back and samples an error point, and nothing
#: further is reached after that. 3 is shipped over 2 for a different, also
#: measured reason: round 3 is the first round whose prompt is built by
#: concatenating onto an ALREADY-concatenated point set, so it is the first
#: round that exercises the accumulation path rather than the initial
#: concatenation. Per-step GPU peak across rounds 1-4 stayed in the 65-104 MiB
#: band at this fixture size with no monotone growth, so the choice costs
#: nothing measurable here. It is NOT a claim about `vit_b` scale.
TRAINING_REFINEMENT_ROUNDS = 3
#: Point label written when the error region a round wanted to sample from is
#: EMPTY. `-1` is `PromptEncoder`'s padding label, whose positional encoding is
#: zeroed (D-013), so an empty region contributes the not-a-point embedding
#: rather than an arbitrary pixel dressed up as a real prompt.
EMPTY_REGION_LABEL = -1
#: Foreground / background labels for the two sampled refinement points.
FOREGROUND_LABEL = 1
BACKGROUND_LABEL = 0
#: Logit written into pixels OUTSIDE the region being sampled from. Large and
#: negative rather than `-inf`: `keras.random.categorical` on an all-`-inf` row
#: returns NaN-driven garbage, while an all-`LOW` row degenerates to a uniform
#: draw that `EMPTY_REGION_LABEL` then neutralizes.
OUTSIDE_REGION_LOGIT = -1e9


def achieved_mask_iou(
    mask_logits: Any,
    gt_masks: Any,
    threshold: float = IOU_MASK_THRESHOLD,
    smooth: float = IOU_SMOOTH,
) -> Any:
    """
    IoU actually achieved by a THRESHOLDED mask prediction against the GT.

    This is the target reference SAM trains the IoU head to predict. Because it
    is computed on the thresholded prediction it carries no gradient with
    respect to the logits; ``SAMTrainingModel.call`` additionally wraps it in
    ``ops.stop_gradient`` so the intent is explicit rather than incidental.

    Args:
        mask_logits: Predicted mask logits, ``(B, M, h, w)``.
        gt_masks: Binary ground truth, ``(B, M, h, w)``.
        threshold: Logit threshold for "foreground".
        smooth: Added to numerator and denominator so an empty union gives
            ``1.0`` rather than ``nan``.

    Returns:
        ``(B, M)`` IoU in ``[0, 1]``.
    """
    predicted = ops.cast(mask_logits > threshold, "float32")
    truth = ops.cast(gt_masks > 0.5, "float32")
    intersection = ops.sum(predicted * truth, axis=[-2, -1])
    union = (
        ops.sum(predicted, axis=[-2, -1])
        + ops.sum(truth, axis=[-2, -1])
        - intersection
    )
    return (intersection + smooth) / (union + smooth)


@keras.saving.register_keras_serializable()
class SAMTrainingModel(keras.Model):
    """
    A trainable ``keras.Model`` that drives :class:`SAM`'s submodules directly.

    ``call`` never invokes ``SAM.call`` / ``SAM.__call__``; that is pinned by a
    spy in ``tests/test_models/test_sam/test_training_model.py``, not by
    inspection.

    Args:
        sam: The :class:`SAM` model to train. Held, not copied; its weights are
            this wrapper's trainable variables.
        multimask_output: Forwarded to ``MaskDecoder``. ``False`` (the default)
            emits a single mask per prompt, which is the unambiguous-prompt
            training regime; ``True`` emits ``num_multimask_outputs`` masks and
            requires a loss that reduces over the mask axis.
        seed: Seed for the ``keras.random.SeedGenerator`` created in
            ``__init__``. The generator is created there, and never lazily
            inside ``call``: this class defines ``build()``, so the layer's
            state tracker is locked before ``call`` runs and a lazily created
            generator raises ``ValueError: You cannot add new elements of state
            (variables or sub-layers) to a layer that is already built``
            (measured; see D-037).
        num_refinement_rounds: How many decode rounds ``call`` runs. ``1``
            disables refinement entirely. Each round after the first is
            prompted by the previous round's **detached** logits (as the dense
            mask prompt) plus two freshly sampled points: a foreground point
            from the false-negative region and a background point from the
            false-positive region. Sampling needs ``gt_mask``; without it the
            rounds still run, prompted by the mask feedback alone.
        **kwargs: Forwarded to ``keras.Model``.

    Input dict (from the ``keras``/data pipeline):
        - ``image``: ``(B, H, W, 3)`` float, values in ``[0, 255]``. ``H`` and
          ``W`` must not exceed ``sam.image_encoder.img_size``; ``preprocess``
          pads, it never resizes (use ``resize_longest_side`` upstream).
        - ``point_coords``: ``(B, N, 2)`` float, in the padded input frame.
        - ``point_labels``: ``(B, N)`` int; ``1`` foreground, ``0`` background,
          ``-1`` padding.
        - ``boxes``: ``(B, K, 4)`` float xyxy, optional.
        - ``gt_mask``: ``(B, M, 4*grid, 4*grid)`` binary, optional. Supplying it
          turns on the ``iou_supervision`` output.

    Output dict (``R = num_refinement_rounds``, rounds concatenated round-major
    on the mask axis):
        - ``low_res_logits``: ``(B, M*R, 4*grid, 4*grid)`` float logits.
        - ``iou_predictions``: ``(B, M*R)`` float.
        - ``iou_supervision``: ``(B, M*R, 2)``, present only when ``gt_mask`` is
          supplied. ``[..., 0]`` is the predicted IoU, ``[..., 1]`` the achieved
          IoU (stop-gradient). ``SAMIoULoss`` reads both from this one tensor.

    Raises:
        ValueError: from ``call`` if ``image`` is absent, if the point keys are
            supplied only half, or if no prompt at all is supplied.
    """

    def __init__(
        self,
        sam: SAM,
        multimask_output: bool = False,
        seed: int = 42,
        num_refinement_rounds: int = DEFAULT_REFINEMENT_ROUNDS,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not isinstance(sam, SAM):
            raise ValueError(
                f"SAMTrainingModel requires a SAM instance, got {type(sam).__name__}"
            )
        self.sam = sam
        self.multimask_output = bool(multimask_output)
        self.seed = int(seed)
        if int(num_refinement_rounds) < 1:
            raise ValueError(
                "num_refinement_rounds must be >= 1 (1 = a single decode with "
                f"no refinement); got {num_refinement_rounds}."
            )
        self.num_refinement_rounds = int(num_refinement_rounds)
        # DECISION plan-2026-08-03T191222-1d751f81/D-037: create the
        # SeedGenerator HERE, in `__init__`. Do NOT create it lazily inside
        # `call()` (`if not hasattr(self, "seed_generator"): ...`), which is the
        # spelling that looks tidier because the generator is only needed when
        # refinement runs. That spelling raises, MEASURED on this class:
        #   ValueError: You cannot add new elements of state (variables or
        #   sub-layers) to a layer that is already built.
        # ...from `Layer._lock_state`, because this class defines `build()`, so
        # `super().build()` locks the tracker before `call` ever runs.
        #
        # F-5 item 10's stronger claim -- that a BARE `keras.random.*` with no
        # seed generator raises the same error -- was RED-probed here and does
        # NOT reproduce on keras 3.8.0: the bare call routes to the module-level
        # global generator, adds no state to this layer, and trains fine. The
        # generator is kept anyway on two measured grounds: the lazy spelling
        # above really does raise, and an explicit per-model generator makes the
        # sampling reproducible from `seed` and serializable, which the global
        # one is not.
        self.seed_generator = keras.random.SeedGenerator(seed=self.seed)

    def build(self, input_shape: Optional[Any] = None) -> None:
        """
        Build the wrapped SAM eagerly, then this model.

        Args:
            input_shape: Unused; forwarded to ``keras.Model.build``.

        Note:
            F-5 item 10 predicted this call would be needed because
            ``PromptEncoder``'s mask-downscaling stack builds LAZILY, so a
            refinement round 2 (the first round with a mask prompt) would raise
            "cannot add new elements of state to a layer that is already
            built". **That prediction was re-measured here and does NOT hold on
            this code**: ``PromptEncoder.build`` explicitly builds
            ``mask_downscaling`` at ``(None, None, None, 1)``, so the path is
            already materialized by an ordinary first forward and a subsequent
            traced mask-prompt call succeeds with or without this line. The
            call is retained on the *other*, still-valid ground -- the same one
            ``SAM.build`` itself documents: it materializes every sub-model
            eagerly, before any weight restore, without requiring a forward
            pass.
        """
        # DECISION plan-2026-08-03T191222-1d751f81/D-035: do NOT delete this
        # line, and do NOT restore the "the mask-prompt path is lazy"
        # justification it used to carry -- that was MEASURED FALSE (see the
        # docstring). What it does buy is that `SAMTrainingModel(...).build(None)`
        # alone leaves the whole SAM built, with no forward pass: `keras.Model.build`
        # on its own leaves `prompt_encoder.built is False` until something calls
        # it. Pinned in both directions by
        # `test_build_alone_materializes_the_whole_sam` and its
        # `keras.Model.build`-only control.
        self.sam.build(None)
        super().build(input_shape)

    # -----------------------------------------------------------------
    # Refinement helpers
    # -----------------------------------------------------------------
    def _feedback_mask(self, low_res_logits: Any) -> Any:
        """
        Turn a round's logits into the next round's dense mask prompt.

        Args:
            low_res_logits: ``(B, M, h, w)`` logits from the round that just
                decoded. Only mask ``0`` is fed back, because ``PromptEncoder``
                accepts exactly one dense mask channel.

        Returns:
            ``(B, 1, h, w)``, detached from the graph.
        """
        # DECISION plan-2026-08-03T191222-1d751f81/D-037: `ops.stop_gradient` is
        # what makes the loop N cheap decodes instead of one N-deep recurrent
        # graph. Do NOT remove it "so the rounds can learn from each other":
        # reference SAM detaches the fed-back mask, and without the detach every
        # round's backward pass re-enters every earlier round's decoder, which
        # is the memory blow-up this loop is designed not to have. Pinned by
        # `TestRefinementStopGradient`, whose control returns this same slice
        # WITHOUT the detach and measures max|g| 0.0 -> non-zero on the image
        # encoder's weights. Note the gradient comes back as exact ZEROS, not
        # `None` -- the surrounding `ops.stack`/`ops.concatenate` keep the
        # tensor structurally connected -- so a `None`-counting assertion here
        # would fail for a reason unrelated to the property (D-036 lesson 1).
        return ops.stop_gradient(low_res_logits[:, 0:1])

    def _sample_from_region(
        self,
        region: Any,
        label: int,
        image_size: Tuple[int, int],
    ) -> Tuple[Any, Any]:
        """
        Sample one pixel per batch row from a binary region.

        Args:
            region: ``(B, h, w)`` float, non-zero inside the region.
            label: Point label to emit when the region is non-empty.
            image_size: ``(H, W)`` of the padded image frame the returned
                coordinates must live in.

        Returns:
            ``(coords, labels)`` of shapes ``(B, 1, 2)`` (xy, float) and
            ``(B, 1)`` (int32). Rows whose region is EMPTY carry
            :data:`EMPTY_REGION_LABEL`.
        """
        shape = tuple(region.shape)
        height, width = int(shape[1]), int(shape[2])
        flat = ops.reshape(region, (-1, height * width))

        # DECISION plan-2026-08-03T191222-1d751f81/D-037: an empty error region
        # is handled by the LABEL, not by the draw. Every pixel outside the
        # region gets `OUTSIDE_REGION_LOGIT`, so an all-empty row degenerates to
        # a uniform draw over the whole grid -- and that draw is then labelled
        # `-1` (padding), whose positional encoding `PromptEncoder` zeroes
        # (D-013). Do NOT "fix" the empty case by clamping the coordinate to
        # (0, 0) with a foreground label: that silently teaches the model that
        # the top-left corner is object interior whenever the prediction is
        # already perfect, and no shape assertion can see it.
        logits = ops.where(
            flat > 0.0,
            ops.zeros_like(flat),
            ops.full_like(flat, OUTSIDE_REGION_LOGIT),
        )
        drawn = keras.random.categorical(logits, 1, seed=self.seed_generator)
        index = ops.cast(ops.squeeze(drawn, axis=-1), "int32")
        row = ops.cast(index // width, "float32")
        col = ops.cast(index - (index // width) * width, "float32")

        # The region grid is `low_res_logits`-sized; the prompt frame is the
        # padded image. Map cell (row, col) to its CENTRE in image pixels, then
        # subtract 0.5 because `PromptEncoder._embed_points` adds its own +0.5
        # pixel-centre offset -- so the encoder ends up at exactly the centre.
        scale_y = float(image_size[0]) / float(height)
        scale_x = float(image_size[1]) / float(width)
        x = (col + 0.5) * scale_x - 0.5
        y = (row + 0.5) * scale_y - 0.5
        coords = ops.expand_dims(ops.stack([x, y], axis=-1), axis=1)

        non_empty = ops.sum(flat, axis=-1) > 0.0
        labels = ops.where(
            non_empty,
            ops.full_like(index, label),
            ops.full_like(index, EMPTY_REGION_LABEL),
        )
        return coords, ops.expand_dims(labels, axis=1)

    def _sample_error_points(
        self,
        low_res_logits: Any,
        gt_mask: Any,
        image_size: Tuple[int, int],
    ) -> Tuple[Any, Any]:
        """
        One foreground point from the false negatives, one background point
        from the false positives.

        Args:
            low_res_logits: ``(B, M, h, w)`` logits of the round that just ran.
            gt_mask: ``(B, M, h, w)`` binary ground truth at the same resolution.
            image_size: ``(H, W)`` of the padded image frame.

        Returns:
            ``(coords, labels)`` of shapes ``(B, 2, 2)`` and ``(B, 2)``.
        """
        predicted = ops.cast(low_res_logits[:, 0] > IOU_MASK_THRESHOLD, "float32")
        truth = ops.cast(gt_mask[:, 0] > 0.5, "float32")
        false_negative = truth * (1.0 - predicted)
        false_positive = (1.0 - truth) * predicted

        fg_coords, fg_labels = self._sample_from_region(
            false_negative, FOREGROUND_LABEL, image_size
        )
        bg_coords, bg_labels = self._sample_from_region(
            false_positive, BACKGROUND_LABEL, image_size
        )
        return (
            ops.concatenate([fg_coords, bg_coords], axis=1),
            ops.concatenate([fg_labels, bg_labels], axis=1),
        )

    def call(
        self,
        inputs: Dict[str, Any],
        training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        """
        Run ``num_refinement_rounds`` decode rounds and stack their outputs.

        The image encoder runs **once**, outside the loop -- that is what makes
        multi-round refinement affordable at all. Each subsequent round re-runs
        only ``prompt_encoder -> mask_decoder``, prompted by the previous
        round's detached logits plus two freshly sampled error points.

        Args:
            inputs: The input dict described in the class docstring.
            training: Standard Keras training flag, forwarded to every submodule.

        Returns:
            ``{"low_res_logits": (B, M*R, h, w), "iou_predictions": (B, M*R)}``
            with ``R = num_refinement_rounds``, rounds concatenated on the mask
            axis in round order, plus ``iou_supervision`` when ``gt_mask`` is
            supplied.

        Raises:
            ValueError: if ``image`` is missing, if exactly one of
                ``point_coords`` / ``point_labels`` is supplied, or if neither
                points nor boxes are supplied (a prompt-less SAM forward is a
                silent no-op the prompt encoder would happily pad).
        """
        if INPUT_IMAGE not in inputs:
            raise ValueError(
                f"SAMTrainingModel input dict must contain '{INPUT_IMAGE}'; got "
                f"keys {sorted(inputs.keys())}"
            )

        has_coords = INPUT_POINT_COORDS in inputs
        has_labels = INPUT_POINT_LABELS in inputs
        if has_coords != has_labels:
            raise ValueError(
                f"'{INPUT_POINT_COORDS}' and '{INPUT_POINT_LABELS}' must be "
                f"supplied together; got keys {sorted(inputs.keys())}"
            )
        boxes = inputs.get(INPUT_BOXES)
        if not has_coords and boxes is None:
            raise ValueError(
                "SAMTrainingModel requires at least one prompt: supply "
                f"'{INPUT_POINT_COORDS}'/'{INPUT_POINT_LABELS}' and/or "
                f"'{INPUT_BOXES}'. A prompt-less forward silently trains the "
                "model to ignore prompts."
            )

        # DECISION plan-2026-08-03T191222-1d751f81/D-035: call the SUBMODULES,
        # never `self.sam(...)` / `self.sam.call(...)`. `SAM.call` ends in
        # `postprocess_masks`, whose `ops.image.resize` raises
        # `TypeError: len is not well defined for a symbolic Tensor` under
        # `fit()`'s graph mode -- unconditionally, no matter which output key is
        # consumed. Do NOT "simplify" this to `self.sam(inputs)` and do NOT
        # reach for the cheaper-looking `self.sam.call(image=..., ...)` bypass:
        # that one does trace, but pays a full-resolution resize plus a `uint8`
        # cast on every training step and skips Keras' own `__call__`
        # bookkeeping. Pinned by `TestSAMCallSpy`, whose control routes through
        # `SAM.call` and observes that exact TypeError.
        image = self.sam.preprocess(inputs[INPUT_IMAGE])
        image_embeddings = self.sam.image_encoder(image, training=training)
        image_size = tuple(self.sam.prompt_encoder.input_image_size)

        coords = inputs[INPUT_POINT_COORDS] if has_coords else None
        labels = inputs[INPUT_POINT_LABELS] if has_coords else None
        gt_mask = inputs.get(INPUT_GT_MASK)
        mask_prompt = None

        # DECISION plan-2026-08-03T191222-1d751f81/D-037: the image encoder is
        # called ONCE, above, and the loop below re-runs only the prompt encoder
        # and the mask decoder. Do NOT move `image_encoder` inside the loop
        # "for symmetry" -- the encoder is >95% of SAM's parameters and the
        # whole feasibility of multi-round refinement on a 12 GB card rests on
        # it running once per step. Do NOT reach for a custom `train_step` to
        # host this loop either: it traces and trains under stock graph-mode
        # `fit()` exactly as written (measured), and a custom `train_step` is
        # forbidden by standing instruction.
        round_logits = []
        round_ious = []
        for round_index in range(self.num_refinement_rounds):
            points = None if coords is None else (coords, labels)
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=points,
                boxes=boxes,
                masks=mask_prompt,
                training=training,
            )
            low_res_logits, iou_predictions = self.sam.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=self.multimask_output,
                training=training,
            )
            round_logits.append(low_res_logits)
            round_ious.append(iou_predictions)

            if round_index + 1 < self.num_refinement_rounds:
                mask_prompt = self._feedback_mask(low_res_logits)
                if gt_mask is not None:
                    new_coords, new_labels = self._sample_error_points(
                        mask_prompt, gt_mask, image_size
                    )
                    if coords is None:
                        coords, labels = new_coords, new_labels
                    else:
                        coords = ops.concatenate(
                            [coords, ops.cast(new_coords, coords.dtype)], axis=1
                        )
                        labels = ops.concatenate(
                            [labels, ops.cast(new_labels, labels.dtype)], axis=1
                        )

        if len(round_logits) == 1:
            all_logits, all_ious = round_logits[0], round_ious[0]
        else:
            all_logits = ops.concatenate(round_logits, axis=1)
            all_ious = ops.concatenate(round_ious, axis=1)

        outputs = {
            LOW_RES_LOGITS: all_logits,
            IOU_PREDICTIONS: all_ious,
        }

        # DECISION plan-2026-08-03T191222-1d751f81/D-036: the IoU target is
        # packed into the SAME tensor as the IoU prediction, and that is forced
        # by the framework, not chosen for convenience. The achieved IoU is a
        # function of the prediction AND the GT together, so it exists only
        # here; a data pipeline cannot produce it, and stock
        # `compile(loss={...})` hands each loss only its own output key. Do NOT
        # "fix" this by (a) supervising `iou_predictions` against a pipeline
        # target -- there is no such target; (b) adding a separate `iou_target`
        # output key -- the loss for `iou_predictions` would still never see it;
        # or (c) writing a custom `train_step`, which is forbidden by standing
        # instruction and unnecessary. The key is emitted only when `gt_mask` is
        # supplied, so an inference-shaped call keeps the two-key contract.
        if gt_mask is not None:
            # DECISION plan-2026-08-03T191222-1d751f81/D-044: the repetition
            # factor is DERIVED from the two shapes by the SAME function the
            # loss uses, never recomputed here. Do NOT "simplify" this back to
            # `[gt_mask] * self.num_refinement_rounds`: the mask axis is
            # `M * R`, not `R`, so at `multimask_output=True` (M=3) and the
            # trainer's default `R=3` that spelling stacked 3 GT copies against
            # 9 logits and died with
            # `ValueError: Dimensions must be equal, but are 9 and 3`. The two
            # halves of one contract must not each own a copy of the
            # derivation. `match_mask_axis` also carries the round-major
            # `concatenate` (not `repeat`, not `tile`) reasoning; see its
            # docstring.
            gt_for_iou = match_mask_axis(
                gt_mask, tuple(gt_mask.shape), tuple(all_logits.shape)
            )
            achieved = ops.stop_gradient(
                achieved_mask_iou(all_logits, gt_for_iou)
            )
            outputs[IOU_SUPERVISION] = ops.stack([all_ious, achieved], axis=-1)

        return outputs

    def get_config(self) -> Dict[str, Any]:
        """
        Serialize the wrapper, including the whole wrapped SAM.

        Returns:
            Configuration dict consumable by :meth:`from_config`.

        Note:
            Without an explicit ``get_config``/``from_config`` pair a ``.keras``
            round-trip of this class fails with
            ``__init__() missing 1 required positional argument: 'sam'``
            (measured, F-5 item 8).
        """
        config = super().get_config()
        config.update(
            {
                "sam": keras.layers.serialize(self.sam),
                "multimask_output": self.multimask_output,
                "seed": self.seed,
                "num_refinement_rounds": self.num_refinement_rounds,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SAMTrainingModel":
        """
        Rebuild a wrapper (and its SAM) from :meth:`get_config` output.

        Args:
            config: A dict produced by :meth:`get_config`.

        Returns:
            A new :class:`SAMTrainingModel`.
        """
        config = dict(config)
        config["sam"] = keras.layers.deserialize(config["sam"])
        return cls(**config)
