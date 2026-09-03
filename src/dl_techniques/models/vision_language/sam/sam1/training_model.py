"""
``SAMTrainingModel``: the trainable, traceable wrapper around :class:`SAM`,
built by :class:`SAMTrainingModel`.

``SAM.call`` cannot be traced under `fit()`'s graph mode -- it always ends
in a resize that raises there -- so this wrapper calls SAM's submodules
directly (``preprocess -> image_encoder -> prompt_encoder -> mask_decoder``)
and returns a dict of differentiable tensors that stock ``compile()``/
``fit()`` can train, without a custom ``train_step``. Multi-round refinement
(``num_refinement_rounds > 1``) re-prompts each round with the previous
round's detached logits plus two freshly sampled error points; the image
encoder still runs only once, outside the loop.

This module makes no accuracy claim: it proves the training path runs with
live gradients, not that SAM trained to any quality, and no official Meta
checkpoint has ever been loaded here. At ``multimask_output=True`` every
proposal is supervised against the same single ground-truth mask (the
paper's minimum-loss-over-masks reduction is not implemented).

References:
    - Kirillov et al., 2023. Segment Anything. (https://arxiv.org/abs/2304.02643)
"""

from typing import Any, Dict, Optional, Tuple

import keras
from keras import ops

from dl_techniques.losses.sam_mask_loss import match_mask_axis
from .model import SAM
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------------
# Public output-key constants. Shared with the losses and the trainer so the
# `compile(loss={...})` dict cannot drift from what `call` actually returns.
# ---------------------------------------------------------------------------
#: Key of the low-resolution mask logits, the training target.
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

#: Rounds the class defaults to; 1 means refinement off. At any value > 1 the
#: mask axis becomes M * rounds, so the wrapper is no longer value-exactly
#: equivalent to an eager SAM.call.
DEFAULT_REFINEMENT_ROUNDS = 1
#: Rounds the trainer ships. Variable coverage on the reduced fixture
#: saturates at 2 rounds (181/201 moved), but 3 is the first round whose
#: prompt accumulates onto an already-concatenated point set.
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
    Compute the IoU a thresholded mask prediction actually achieves against
    the ground truth -- the target reference SAM trains the IoU head to
    predict. Carries no gradient (thresholding is non-differentiable);
    ``SAMTrainingModel.call`` wraps the result in ``stop_gradient`` as well.

    :param mask_logits: Predicted mask logits, ``(B, M, h, w)``.
    :param gt_masks: Binary ground truth, ``(B, M, h, w)``.
    :param threshold: Logit threshold for "foreground".
    :type threshold: float
    :param smooth: Added to numerator and denominator so an empty union
        gives ``1.0`` rather than ``nan``.
    :type smooth: float
    :return: IoU in ``[0, 1]``, shape ``(B, M)``.
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


@register_dl_technique("dl_techniques.models.sam1.training_model")
class SAMTrainingModel(keras.Model):
    """
    A trainable model that drives :class:`SAM`'s submodules directly instead
    of calling ``SAM.call``.

    Architecture:

    .. code-block:: text

        image ─► preprocess ─► image_encoder ─► image_embeddings (once)
                                                        │
        points/boxes ──────────────────────────┐        │
        (mask_prompt from previous round) ──────┤        │
                                                 ▼        ▼
                                          prompt_encoder  │
                                                 │        │
                                                 ▼        ▼
                                             mask_decoder
                                                 │
                                    ┌────────────┴────────────┐
                                    ▼                          ▼
                          low_res_logits, iou_predictions   (repeat num_refinement_rounds
                                    │                         times, feeding logits back)
                                    ▼
                    concatenate round-major on mask axis
                                    │
                    (if gt_mask) ──►│──► iou_supervision
                                    ▼
                                outputs

    :param sam: The :class:`SAM` model to train. Held, not copied; its
        weights are this wrapper's trainable variables.
    :type sam: SAM
    :param multimask_output: Forwarded to the mask decoder. False (the
        default) emits a single mask per prompt; True emits
        ``num_multimask_outputs`` masks and requires a loss that reduces
        over the mask axis.
    :type multimask_output: bool
    :param seed: Seed for the refinement-sampling ``SeedGenerator``, created
        in ``__init__`` rather than lazily in ``call`` (see the DECISION
        comment there).
    :type seed: int
    :param num_refinement_rounds: Number of decode rounds. 1 disables
        refinement. Each later round is prompted by the previous round's
        detached logits plus two freshly sampled error points (needs
        ``gt_mask``; without it the rounds still run on mask feedback alone).
    :type num_refinement_rounds: int
    :param kwargs: Forwarded to ``keras.Model``.
    :raises ValueError: If ``sam`` is not a :class:`SAM` instance, or
        ``num_refinement_rounds < 1``.

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

    .. note::
       ``call`` itself also raises ``ValueError`` if ``image`` is absent, if
       the point keys are supplied only half, or if no prompt is supplied.
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
        # SeedGenerator here, in __init__, not lazily in call() -- this class
        # defines build(), so the tracker is locked before call() runs and a lazy generator raises. See decisions.md.
        self.seed_generator = keras.random.SeedGenerator(seed=self.seed)

    def build(self, input_shape: Optional[Any] = None) -> None:
        """
        Build the wrapped SAM eagerly, then this model.

        :param input_shape: Unused; forwarded to ``keras.Model.build``.
        :type input_shape: Optional[Any]

        .. note::
           DECISION plan-2026-08-03T191222-1d751f81/D-035: this call is what
           makes ``SAMTrainingModel(...).build(None)`` alone leave the whole
           SAM built, with no forward pass -- ``keras.Model.build`` on its own
           leaves ``prompt_encoder.built`` False. See decisions.md.
        """
        self.sam.build(None)
        super().build(input_shape)

    # -----------------------------------------------------------------
    # Refinement helpers
    # -----------------------------------------------------------------
    def _feedback_mask(self, low_res_logits: Any) -> Any:
        """
        Turn a round's logits into the next round's dense mask prompt.

        :param low_res_logits: Logits from the round that just decoded,
            shape ``(B, M, h, w)``. Only mask 0 is fed back, since
            ``PromptEncoder`` accepts exactly one dense mask channel.
        :return: Detached mask prompt, shape ``(B, 1, h, w)``.
        """
        # DECISION plan-2026-08-03T191222-1d751f81/D-037: stop_gradient here
        # is what makes the loop N cheap decodes instead of one N-deep
        # recurrent graph -- without it every round re-enters every earlier round's decoder. See decisions.md.
        return ops.stop_gradient(low_res_logits[:, 0:1])

    def _sample_from_region(
        self,
        region: Any,
        label: int,
        image_size: Tuple[int, int],
    ) -> Tuple[Any, Any]:
        """
        Sample one pixel per batch row from a binary region.

        :param region: Non-zero inside the region, shape ``(B, h, w)``.
        :param label: Point label to emit when the region is non-empty.
        :type label: int
        :param image_size: ``(H, W)`` of the padded image frame the
            returned coordinates must live in.
        :type image_size: Tuple[int, int]
        :return: ``(coords, labels)`` of shapes ``(B, 1, 2)`` (xy, float)
            and ``(B, 1)`` (int32). Rows whose region is empty carry
            :data:`EMPTY_REGION_LABEL`.
        """
        shape = tuple(region.shape)
        height, width = int(shape[1]), int(shape[2])
        flat = ops.reshape(region, (-1, height * width))

        # DECISION plan-2026-08-03T191222-1d751f81/D-037: an empty error
        # region is handled by the label, not the draw -- an all-empty row
        # degenerates to a uniform draw labelled -1 (padding), whose PE is zeroed (D-013), rather than a clamped fake foreground point. See decisions.md.
        logits = ops.where(
            flat > 0.0,
            ops.zeros_like(flat),
            ops.full_like(flat, OUTSIDE_REGION_LOGIT),
        )
        drawn = keras.random.categorical(logits, 1, seed=self.seed_generator)
        index = ops.cast(ops.squeeze(drawn, axis=-1), "int32")
        row = ops.cast(index // width, "float32")
        col = ops.cast(index - (index // width) * width, "float32")

        # Map cell (row, col) to its centre in image pixels, then subtract 0.5
        # since PromptEncoder._embed_points adds its own +0.5 pixel-centre offset.
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

        :param low_res_logits: Logits of the round that just ran, shape
            ``(B, M, h, w)``.
        :param gt_mask: Binary ground truth at the same resolution, shape
            ``(B, M, h, w)``.
        :param image_size: ``(H, W)`` of the padded image frame.
        :type image_size: Tuple[int, int]
        :return: ``(coords, labels)`` of shapes ``(B, 2, 2)`` and ``(B, 2)``.
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

        The image encoder runs once, outside the loop; each round after the
        first re-runs only ``prompt_encoder -> mask_decoder``, prompted by
        the previous round's detached logits plus two sampled error points.

        :param inputs: The input dict described in the class docstring.
        :type inputs: Dict[str, Any]
        :param training: Standard Keras training flag, forwarded to every
            submodule.
        :type training: Optional[bool]
        :return: ``{"low_res_logits": (B, M*R, h, w), "iou_predictions": (B,
            M*R)}`` with ``R = num_refinement_rounds``, rounds concatenated
            round-major on the mask axis, plus ``iou_supervision`` when
            ``gt_mask`` is supplied.
        :rtype: Dict[str, keras.KerasTensor]
        :raises ValueError: If ``image`` is missing, if exactly one of
            ``point_coords``/``point_labels`` is supplied, or if neither
            points nor boxes are supplied.
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

        # DECISION plan-2026-08-03T191222-1d751f81/D-035: call the submodules,
        # never self.sam(...)/self.sam.call(...) -- SAM.call always ends in
        # postprocess_masks, whose resize raises under fit()'s graph mode regardless of output key. See decisions.md.
        image = self.sam.preprocess(inputs[INPUT_IMAGE])
        image_embeddings = self.sam.image_encoder(image, training=training)
        image_size = tuple(self.sam.prompt_encoder.input_image_size)

        coords = inputs[INPUT_POINT_COORDS] if has_coords else None
        labels = inputs[INPUT_POINT_LABELS] if has_coords else None
        gt_mask = inputs.get(INPUT_GT_MASK)
        mask_prompt = None

        # DECISION plan-2026-08-03T191222-1d751f81/D-037: the image encoder
        # runs once, above; the loop below re-runs only the prompt encoder and
        # mask decoder -- the encoder is >95% of SAM's parameters, and multi-round refinement on a 12 GB card depends on running it once per step. See decisions.md.
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
        # packed into the same tensor as the prediction, forced by the
        # framework -- the achieved IoU exists only here (needs prediction and GT together) and compile(loss={...}) hands each loss only its own key. See decisions.md.
        if gt_mask is not None:
            # DECISION plan-2026-08-03T191222-1d751f81/D-044: the repetition
            # factor comes from match_mask_axis, never recomputed here -- the
            # mask axis is M*R, not R, so `[gt_mask] * rounds` broke at M=3/R=3. See decisions.md.
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

        :return: Configuration dict consumable by :meth:`from_config`.
        :rtype: Dict[str, Any]
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
        Rebuild a wrapper, and its SAM, from :meth:`get_config` output.

        :param config: A dict produced by :meth:`get_config`.
        :type config: Dict[str, Any]
        :return: A new :class:`SAMTrainingModel`.
        :rtype: SAMTrainingModel
        """
        config = dict(config)
        config["sam"] = keras.layers.deserialize(config["sam"])
        return cls(**config)
