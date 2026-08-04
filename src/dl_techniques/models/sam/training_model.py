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
            ``__init__``. The generator is created here, and never inside
            ``call``, because ``keras.random.*`` without an explicit seed
            generator raises ``ValueError: You cannot add new elements of state
            ... to a layer that is already built`` on the second traced call.
        **kwargs: Forwarded to ``keras.Model``.

    Input dict (from ``tf.data``):
        - ``image``: ``(B, H, W, 3)`` float, values in ``[0, 255]``. ``H`` and
          ``W`` must not exceed ``sam.image_encoder.img_size``; ``preprocess``
          pads, it never resizes (use ``resize_longest_side`` upstream).
        - ``point_coords``: ``(B, N, 2)`` float, in the padded input frame.
        - ``point_labels``: ``(B, N)`` int; ``1`` foreground, ``0`` background,
          ``-1`` padding.
        - ``boxes``: ``(B, K, 4)`` float xyxy, optional.
        - ``gt_mask``: ``(B, M, 4*grid, 4*grid)`` binary, optional. Supplying it
          turns on the ``iou_supervision`` output.

    Output dict:
        - ``low_res_logits``: ``(B, M, 4*grid, 4*grid)`` float logits.
        - ``iou_predictions``: ``(B, M)`` float.
        - ``iou_supervision``: ``(B, M, 2)``, present only when ``gt_mask`` is
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
        # DECISION plan-2026-08-03T191222-1d751f81/D-035: create the
        # SeedGenerator HERE, in `__init__`, even though nothing in this class
        # samples yet (the refinement loop, step 4, will). Do NOT call a bare
        # `keras.random.*` inside `call()` and do NOT lazily create the
        # generator on first use: Keras creates the generator's state variable
        # at first call, and a layer that is already built refuses new state
        # with `ValueError: You cannot add new elements of state ... to a layer
        # that is already built`. Measured (F-5 item 10) -- the failure appears
        # on the SECOND traced call, so a single-call smoke test cannot see it.
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

    def call(
        self,
        inputs: Dict[str, Any],
        training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        """
        Run ``preprocess -> image_encoder -> prompt_encoder -> mask_decoder``.

        Args:
            inputs: The input dict described in the class docstring.
            training: Standard Keras training flag, forwarded to every submodule.

        Returns:
            ``{"low_res_logits": ..., "iou_predictions": ...}``.

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

        points = None
        if has_coords:
            points = (inputs[INPUT_POINT_COORDS], inputs[INPUT_POINT_LABELS])

        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=None,
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

        outputs = {
            LOW_RES_LOGITS: low_res_logits,
            IOU_PREDICTIONS: iou_predictions,
        }

        # DECISION plan-2026-08-03T191222-1d751f81/D-036: the IoU target is
        # packed into the SAME tensor as the IoU prediction, and that is forced
        # by the framework, not chosen for convenience. The achieved IoU is a
        # function of the prediction AND the GT together, so it exists only
        # here; a `tf.data` pipeline cannot produce it, and stock
        # `compile(loss={...})` hands each loss only its own output key. Do NOT
        # "fix" this by (a) supervising `iou_predictions` against a pipeline
        # target -- there is no such target; (b) adding a separate `iou_target`
        # output key -- the loss for `iou_predictions` would still never see it;
        # or (c) writing a custom `train_step`, which is forbidden by standing
        # instruction and unnecessary. The key is emitted only when `gt_mask` is
        # supplied, so an inference-shaped call keeps the two-key contract.
        if INPUT_GT_MASK in inputs:
            achieved = ops.stop_gradient(
                achieved_mask_iou(low_res_logits, inputs[INPUT_GT_MASK])
            )
            outputs[IOU_SUPERVISION] = ops.stack(
                [iou_predictions, achieved], axis=-1
            )

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
