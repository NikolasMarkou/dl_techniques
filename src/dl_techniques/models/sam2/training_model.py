"""``SAM2TrainingModel`` -- the trainable, traceable multi-frame wrapper.

Why a wrapper exists at all
---------------------------
:meth:`SAM2.stream_step` is the video path, and it is deliberately NOT
traceable: it mutates a Python object (the memory bank), branches on whether
that object is empty, and reads Python integers out of its selection policy.
It therefore cannot be the inner operation of a ``fit()`` step, and a custom
``train_step`` is forbidden by standing instruction.

This wrapper takes the other route. It runs the image encoder ONCE over a
flattened ``(B * T, ...)`` batch, then an explicit **unrolled Python loop** over
a STATIC ``num_frames``, driving SAM 2's submodules directly:

    ``image_encoder -> memory_attention -> prompt_encoder -> mask_decoder ->
    memory_encoder``

The whole loop traces under stock graph-mode ``fit()``. The memory bank is
constructed FRESH inside :meth:`call`, as a local variable, so it holds no
symbolic tensor across traces and the full iteration-1 selection policy
(``select_frames``, ``select_object_pointer_frames``, the ``t_pos`` slots) is
reused verbatim rather than re-derived. That this traces at all was PROBED
before this module was written; see ``decisions.md`` D-053.

``SAM2.call`` / ``SAM2.__call__`` is never invoked. That is pinned by a spy in
``tests/test_models/test_sam2/test_training_model.py``, not by inspection.

``compile(jit_compile=False)`` IS MANDATORY
-------------------------------------------
Keras 3.8's ``fit()`` defaults to ``jit_compile='auto'``, which selects XLA on
a GPU. ``Hiera``'s stem interpolates its learned positional embedding with a
BICUBIC ``ops.image.resize``, and MEASURED on this stack that op has no XLA GPU
kernel::

    InvalidArgumentError: Detected unsupported operations when trying to
    compile graph ... on XLA_GPU_JIT: ResizeBicubic (No registered
    'ResizeBicubic' OpKernel for XLA_GPU_JIT devices ...)

The failure is at the FIRST ``fit()`` step, is loud, and is not a defect in
this wrapper -- the same graph traces and runs perfectly under a plain
``tf.function``. Every caller that trains this model must pass
``jit_compile=False`` to :meth:`compile`. Pinned in both directions by
``TestXLARefusal``.

This module makes **no accuracy claim**. It proves the multi-frame training
path runs with live gradients; no Meta SAM 2 checkpoint has ever been loaded in
this repository.
"""

from typing import Any, Dict, List, Optional, Tuple

import keras
from keras import ops

from ...losses.sam2_video_loss import SAM2GatedMaskLoss, mask_presence_gate
from ...losses.sam_mask_loss import SAMIoULoss
from ..sam.training_model import achieved_mask_iou
from .memory_bank import SAM2MemoryBank
from .model import SAM2, _select_best_by_iou

# ---------------------------------------------------------------------
# Public output-key constants. Shared with the losses and the trainer so a
# `compile(loss={...})` dict cannot drift from what `call` actually returns
# (H-5: a dict `y_pred` requires `loss=` keyed to the output names).
# ---------------------------------------------------------------------
#: Key of the per-frame low-resolution mask logits, ``(B, T, h, w)``.
SAM2_LOW_RES_LOGITS = "low_res_logits"
#: Key of the per-frame object-score logits, ``(B, T, 1)``. This output is the
#: ONLY differentiable consumer of the object-score head: every other consumer
#: in this package thresholds it hard at ``> 0``, so a trainer that does not
#: put a loss on this key trains a permanently frozen occlusion head.
SAM2_OBJECT_SCORE_LOGITS = "object_score_logits"
#: Key of the packed IoU supervision pair, ``(B, T, 2)``: ``[..., 0]`` is the
#: model's predicted IoU and ``[..., 1]`` the IoU it actually achieved against
#: the ground truth, already ``stop_gradient``-ed. Both channels are ZEROED on
#: rows the ground truth calls absent, which is what makes ``SAMIoULoss``
#: reusable unchanged (D-052): ``zero - zero`` contributes exactly ``0`` to the
#: mean and exactly ``0`` to the gradient, reproducing upstream's
#: ``loss_iou * target_obj``.
SAM2_IOU_SUPERVISION = "iou_supervision"
#: Keys :meth:`SAM2TrainingModel.call` always returns, in a stable order.
OUTPUT_KEYS: Tuple[str, ...] = (
    SAM2_LOW_RES_LOGITS, SAM2_OBJECT_SCORE_LOGITS, SAM2_IOU_SUPERVISION)

#: Input key of the clip, ``(B, T, image_size, image_size, 3)``.
INPUT_IMAGE = "image"
#: Input key of the frame-0 point prompt coordinates, ``(B, N, 2)``.
INPUT_POINT_COORDS = "point_coords"
#: Input key of the frame-0 point prompt labels, ``(B, N)``.
INPUT_POINT_LABELS = "point_labels"
#: Input key of the optional frame-0 box prompt, ``(B, K, 4)`` xyxy.
INPUT_BOXES = "boxes"
#: Input key of the per-frame binary ground truth, ``(B, T, h, w)``. REQUIRED:
#: :data:`SAM2_IOU_SUPERVISION` is computed from it, and the presence gate that
#: zeroes it is derived from it too.
INPUT_GT_MASKS = "gt_masks"


@keras.saving.register_keras_serializable()
class SAM2TrainingModel(keras.Model):
    """A trainable ``keras.Model`` that drives :class:`SAM2`'s submodules.

    :param sam2: The :class:`SAM2` model to train. Held, not copied; its
        weights are this wrapper's trainable variables. Its
        ``multimask_output`` must be ``False``.
    :type sam2: SAM2
    :param num_frames: Clip length ``T``. A **static Python int**, because it
        is the bound of an unrolled Python loop; a symbolic ``T`` would need a
        ``while_loop`` and a memory bank that is not a Python object.
    :type num_frames: int
    :param seed: Reserved for future stochastic prompting; stored and
        serialized so the config is stable across steps.
    :type seed: int
    :param kwargs: Forwarded to ``keras.Model``.

    :raises ValueError: If ``sam2`` is not a :class:`SAM2`, if ``num_frames``
        is below 1, or if ``sam2.multimask_output`` is ``True``.

    Input dict:

    * ``image``: ``(B, T, image_size, image_size, 3)`` float in ``[0, 255]``.
    * ``point_coords`` / ``point_labels``: the FRAME-0 prompt only. Frames
      ``1 .. T-1`` are conditioned by the memory bank, never by a fresh prompt.
    * ``boxes``: optional frame-0 box prompt.
    * ``gt_masks``: optional ``(B, T, h, w)`` binary ground truth.

    Output dict, batch-axis preserving, with the frame axis folded INTO the
    mask axis (D-051) so ``match_mask_axis`` and both ``sam_mask_loss``
    adapters work unmodified:

    * ``low_res_logits``: ``(B, T, h, w)`` -- ``M == 1``, so ``T`` is
      unambiguous on that axis.
    * ``object_score_logits``: ``(B, T, 1)``.
    * ``iou_supervision``: ``(B, T, 2)`` -- see :data:`SAM2_IOU_SUPERVISION`.

    Example:

    .. code-block:: python

        trainer = SAM2TrainingModel(create_sam2("tiny"), num_frames=4)
        compile_sam2_video_trainer(trainer)
        trainer.fit(dataset)
    """

    def __init__(
            self,
            sam2: SAM2,
            num_frames: int = 4,
            seed: int = 42,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not isinstance(sam2, SAM2):
            raise ValueError(
                "SAM2TrainingModel requires a SAM2 instance, got "
                f"{type(sam2).__name__}"
            )
        if int(num_frames) < 1:
            raise ValueError(
                f"num_frames must be >= 1 (1 degenerates to the image path); "
                f"got {num_frames}."
            )
        # DECISION plan-2026-08-04T044628-4c240b4c/D-051
        # `multimask_output=True` is REFUSED, not supported. Do NOT "add
        # support" by widening the mask axis: this wrapper folds the FRAME axis
        # into the mask axis, so at M > 1 `low_res_logits` would be `(B, T*M,
        # h, w)` interleaved frame-major (f0m0, f0m1, f0m2, f1m0, ...) while
        # `match_mask_axis` repeats the ground truth ROUND-major
        # ([f0..fT, f0..fT, f0..fT]). The two orders do not align, every shape
        # still checks out, and nothing anywhere raises -- the loss simply
        # supervises frame 1's mask against frame 0's ground truth. Upstream
        # additionally selects the supervised multimask slice by
        # `argmin(20 * loss_mask + loss_dice)` against the GT, not by predicted
        # IoU, so a correct multimask training path needs a different selector
        # entirely. See decisions.md D-051.
        if bool(sam2.multimask_output):
            raise ValueError(
                "SAM2TrainingModel refuses multimask_output=True. The frame "
                "axis is folded into the mask axis, so at M > 1 'low_res_"
                "logits' is (B, T*M, h, w) interleaved FRAME-major while "
                "match_mask_axis repeats the ground truth ROUND-major; the "
                "two do not align and no shape check can see it. Build the "
                "model with create_sam2(..., multimask_output=False)."
            )

        self.sam2 = sam2
        self.num_frames = int(num_frames)
        self.seed = int(seed)

    def build(self, input_shape: Optional[Any] = None) -> None:
        """Build the wrapped SAM 2 eagerly, then this model.

        :param input_shape: Unused; forwarded to ``keras.Model.build``.
        :type input_shape: Optional[Any]
        """
        # SAM 1's precedent (its D-035's still-valid half): this materializes
        # every sub-model without a forward pass, so a weight restore lands on
        # a complete variable set.
        self.sam2.build(None)
        super().build(input_shape)

    # -----------------------------------------------------------------
    # per-frame pieces
    # -----------------------------------------------------------------

    def _unflatten(self, tensor: Any) -> Any:
        """Split a ``(B * T, ...)`` encoder output back into ``(B, T, ...)``.

        :param tensor: A level tensor produced by the single flattened image
            encoder call. Its trailing dimensions must be static.
        :type tensor: Any
        :return: The same tensor with the leading axis split.
        :rtype: Any
        """
        trailing = tuple(int(dim) for dim in tensor.shape[1:])
        return ops.reshape(tensor, (-1, self.num_frames) + trailing)

    def _condition(
            self,
            bank: SAM2MemoryBank,
            features: Any,
            features_pos: Any,
            frame_idx: int,
            training: Optional[bool],
    ) -> Any:
        """Condition one frame's features on the bank, or on ``no_mem_embed``.

        This mirrors :meth:`SAM2._condition_on_memory` exactly, differing only
        in reading a CALLER-SUPPLIED bank instead of ``self.sam2.memory_bank``.

        # DECISION plan-2026-08-04T044628-4c240b4c/D-054
        # This method is a deliberate near-duplicate of
        # `SAM2._condition_on_memory`, and the duplication cannot be removed
        # from this side. That method reads `self.memory_bank` -- the per-VIDEO
        # streaming instance -- while training needs a bank that is local to
        # one traced `call()`. Do NOT "de-duplicate" by assigning a fresh bank
        # onto `self.sam2.memory_bank` before the loop: that mutates shared
        # model state from inside a traced forward pass, so two concurrent
        # traces, or an interleaved `stream_step`, would silently share one
        # bank. The tail of the decode is NOT duplicated -- `_decode` is called
        # directly. See decisions.md D-054.

        :param bank: The per-call memory bank.
        :type bank: SAM2MemoryBank
        :param features: ``(B, grid, grid, hidden_dim)`` raw pixel features.
        :type features: Any
        :param features_pos: Positional encoding of ``features``.
        :type features_pos: Any
        :param frame_idx: The frame being decoded.
        :type frame_idx: int
        :param training: Keras training flag.
        :type training: Optional[bool]
        :return: ``(B, grid, grid, hidden_dim)`` conditioned features.
        :rtype: Any
        """
        hidden = self.sam2.hidden_dim
        grid_h, grid_w = int(features.shape[1]), int(features.shape[2])
        batch = ops.shape(features)[0]
        tokens = ops.reshape(features, (batch, grid_h * grid_w, hidden))
        tokens_pos = ops.reshape(features_pos, (batch, grid_h * grid_w, hidden))

        readout = bank.read(frame_idx)

        if readout.memory is None:
            if self.sam2.directly_add_no_mem_embed:
                conditioned = tokens + ops.cast(
                    self.sam2.no_mem_embed, tokens.dtype)
                return ops.reshape(
                    conditioned, (batch, grid_h, grid_w, hidden))
            memory = ops.zeros(
                (batch, 1, self.sam2.mem_dim), dtype=tokens.dtype)
            memory_pos = ops.broadcast_to(
                ops.cast(self.sam2.no_mem_pos_enc, tokens.dtype),
                (batch, 1, self.sam2.mem_dim),
            )
            num_ptr_tokens = 0
        else:
            # H-4: every fed-back boundary is detached. Without this, a T-frame
            # clip is ONE T-deep recurrent graph instead of T decodes. The bank
            # detaches on insertion too, so the boundary is two-sided.
            memory = ops.stop_gradient(readout.memory)
            memory_pos = ops.stop_gradient(readout.memory_pos)
            memory_pos = memory_pos + self.sam2._temporal_embedding(
                readout, memory)
            num_ptr_tokens = readout.num_obj_ptr_tokens

        conditioned = self.sam2.memory_attention(
            tokens,
            memory,
            features_pos=tokens_pos,
            memory_pos=memory_pos,
            num_obj_ptr_tokens=num_ptr_tokens,
            training=training,
        )
        return ops.reshape(conditioned, (batch, grid_h, grid_w, hidden))

    def _store(
            self,
            bank: SAM2MemoryBank,
            frame_idx: int,
            features: Any,
            outputs: Dict[str, Any],
            training: Optional[bool],
    ) -> None:
        """Encode this frame's prediction and push it into the local bank.

        Mirrors :meth:`SAM2._store_memory`, again differing only in the bank it
        writes to (D-054). The best-IoU selection goes through the SHARED
        :func:`_select_best_by_iou`, and the occlusion mark through
        :meth:`SAM2._mark_occlusion`, so neither semantic is re-derived here.

        :param bank: The per-call memory bank.
        :type bank: SAM2MemoryBank
        :param frame_idx: The frame just decoded.
        :type frame_idx: int
        :param features: ``(B, grid, grid, hidden_dim)`` RAW pixel features --
            not the memory-conditioned ones.
        :type features: Any
        :param outputs: :meth:`SAM2._decode`'s output dict for this frame.
        :type outputs: Dict[str, Any]
        :param training: Keras training flag.
        :type training: Optional[bool]
        """
        logits = _select_best_by_iou(
            outputs["low_res_logits"], outputs["iou_predictions"])
        logits = ops.transpose(logits, (0, 2, 3, 1))
        size = self.sam2.image_size
        high_res = ops.image.resize(
            logits, (size, size), interpolation="bilinear")

        memory, memory_pos = self.sam2.memory_encoder(
            [ops.stop_gradient(features), ops.stop_gradient(high_res)],
            training=training,
        )
        memory = self.sam2._mark_occlusion(
            memory, outputs["object_score_logits"])
        bank.add_frame(
            frame_idx,
            maskmem_features=memory,
            maskmem_pos_enc=memory_pos,
            obj_ptr=ops.stop_gradient(outputs["object_pointer"]),
            is_conditioning=(frame_idx == 0),
        )

    def _iou_supervision(
            self,
            outputs: Dict[str, Any],
            gt_frame: Any,
    ) -> Any:
        """Pack one frame's ``(predicted, achieved)`` IoU pair, gated.

        # DECISION plan-2026-08-04T044628-4c240b4c/D-052
        # BOTH channels are zeroed on a GT-absent row, and `SAMIoULoss` is
        # reused UNCHANGED. Do NOT "fix" this by packing a third presence
        # channel and writing a `SAM2IoULoss`: `SAMIoULoss` computes
        # `mean(square(predicted - achieved))`, so zeroing both sides makes an
        # absent row contribute exactly 0 to the mean AND exactly 0 to the
        # gradient -- which is upstream's `loss_iou * target_obj` exactly, not
        # an approximation of it (plan assumption A-5, hand-verified in
        # `tests/test_losses/test_sam2_video_loss.py`).
        #
        # The hazard this carries, named here because it is invisible at the
        # call site: zeroing both sides of a comparison makes `zero == zero`
        # always agree, so NO liveness probe on this output can discriminate a
        # correct gate from a dead one. Any guard over it must assert the loss
        # VALUE against a hand-computed gated number. And the gate must come
        # from `gt_frame` -- gating on `outputs["object_score_logits"]` would
        # be self-fulfilling and would additionally couple the IoU head's
        # supervision to a head that is itself being trained.
        # See decisions.md D-052.

        :param outputs: :meth:`SAM2._decode`'s output dict for this frame.
        :type outputs: Dict[str, Any]
        :param gt_frame: ``(B, 1, h, w)`` ground truth for this frame -- the
            frame axis kept as a length-1 mask axis so it lines up with
            ``low_res_logits`` without a reshape.
        :type gt_frame: Any
        :return: ``(B, 1, 2)``; concatenating over frames gives ``(B, T, 2)``.
        :rtype: Any
        """
        logits = outputs["low_res_logits"]
        truth = ops.cast(gt_frame, logits.dtype)
        predicted = ops.cast(outputs["iou_predictions"], logits.dtype)
        # `achieved_mask_iou` is IMPORTED from SAM 1's training model, not
        # re-derived: it is public, frame-agnostic and the single source of
        # truth for "the IoU this thresholded prediction actually got". It
        # thresholds, so it is already gradient-free; the explicit
        # `stop_gradient` states the intent rather than relying on that.
        achieved = ops.stop_gradient(achieved_mask_iou(logits, truth))
        packed = ops.stack([predicted, achieved], axis=-1)
        present = ops.reshape(mask_presence_gate(truth), (-1, 1, 1))
        return ops.where(present, packed, ops.zeros_like(packed))

    # -----------------------------------------------------------------
    # the traced forward
    # -----------------------------------------------------------------

    def call(
            self,
            inputs: Dict[str, Any],
            training: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Decode a whole clip, frame by frame, under one traced graph.

        :param inputs: The input dict described in the class docstring.
        :type inputs: Dict[str, Any]
        :param training: Standard Keras training flag, forwarded to every
            submodule.
        :type training: Optional[bool]
        :return: ``{'low_res_logits': (B, T, h, w),
            'object_score_logits': (B, T, 1)}``.
        :rtype: Dict[str, Any]
        :raises ValueError: If ``image`` is absent, if the point keys are
            supplied only half, or if no prompt at all is supplied.
        """
        if INPUT_IMAGE not in inputs:
            raise ValueError(
                f"SAM2TrainingModel input dict must contain '{INPUT_IMAGE}'; "
                f"got keys {sorted(inputs.keys())}"
            )
        has_coords = INPUT_POINT_COORDS in inputs
        has_labels = INPUT_POINT_LABELS in inputs
        if has_coords != has_labels:
            raise ValueError(
                f"'{INPUT_POINT_COORDS}' and '{INPUT_POINT_LABELS}' must be "
                f"supplied together; got keys {sorted(inputs.keys())}"
            )
        if INPUT_GT_MASKS not in inputs:
            raise ValueError(
                f"SAM2TrainingModel input dict must contain '{INPUT_GT_MASKS}' "
                f"-- (B, T, h, w) binary ground truth; got keys "
                f"{sorted(inputs.keys())}. It is REQUIRED, not optional: the "
                f"'{SAM2_IOU_SUPERVISION}' output packs the IoU this "
                f"prediction achieved against it, and the ground-truth "
                f"presence gate that zeroes absent rows is derived from it. "
                f"Emitting zeros instead would leave the IoU head with a "
                f"silently dead loss."
            )
        boxes = inputs.get(INPUT_BOXES)
        if not has_coords and boxes is None:
            raise ValueError(
                "SAM2TrainingModel requires a frame-0 prompt: supply "
                f"'{INPUT_POINT_COORDS}'/'{INPUT_POINT_LABELS}' and/or "
                f"'{INPUT_BOXES}'. A prompt-less clip silently trains the "
                "model to ignore prompts."
            )

        # The encoder runs ONCE, over a flattened (B*T, S, S, 3) batch. Do NOT
        # move it inside the loop "for symmetry": it is the overwhelming
        # majority of SAM 2's compute, and running it per frame is what makes a
        # T-frame clip unaffordable. This mirrors SAM 1's own
        # encoder-outside-the-loop precedent.
        size = self.sam2.image_size
        flat = ops.reshape(inputs[INPUT_IMAGE], (-1, size, size, 3))
        encoded = self.sam2.image_encoder(flat, training=training)

        vision_features = self._unflatten(encoded["vision_features"])
        vision_pos = [self._unflatten(p) for p in encoded["vision_pos_enc"]]
        backbone_fpn = [self._unflatten(f) for f in encoded["backbone_fpn"]]

        # DECISION plan-2026-08-04T044628-4c240b4c/D-053
        # The bank is constructed FRESH here, as a local, on every call. Do NOT
        # hoist it to `__init__` or reuse `self.sam2.memory_bank`: a bank that
        # outlives one `call()` holds tensors from a previous trace, so every
        # frame > 0 would attend to a stale graph's memory -- with no shape,
        # dtype or finiteness symptom. That it traces at all as a per-call
        # local was PROBED before this file existed (outcome (a): one trace,
        # no retrace, outputs vary with inputs). See decisions.md D-053.
        bank = SAM2MemoryBank(
            num_maskmem=self.sam2.num_maskmem,
            mem_dim=self.sam2.mem_dim,
            hidden_dim=self.sam2.hidden_dim,
            memory_temporal_stride_for_eval=(
                self.sam2.memory_temporal_stride_for_eval),
            max_obj_ptrs_in_encoder=self.sam2.max_obj_ptrs_in_encoder,
        )

        coords = inputs[INPUT_POINT_COORDS] if has_coords else None
        labels = inputs[INPUT_POINT_LABELS] if has_coords else None

        gt_masks = inputs[INPUT_GT_MASKS]

        frame_logits: List[Any] = []
        frame_scores: List[Any] = []
        frame_iou: List[Any] = []
        for t in range(self.num_frames):
            features = vision_features[:, t]
            features_pos = vision_pos[-1][:, t]
            conditioned = self._condition(
                bank, features, features_pos, t, training)

            # The prompt is FRAME-0-ONLY. Frames > 0 are conditioned by the
            # memory bank, exactly as `stream_step` conditions them; passing
            # the frame-0 prompt again would teach the model that the object
            # never moves away from its frame-0 location.
            if t == 0:
                prompt = {
                    "points": None if coords is None else (coords, labels),
                    "boxes": boxes,
                    "masks": None,
                }
            else:
                prompt = {"points": None, "boxes": None, "masks": None}

            per_frame_encoded = {
                "vision_features": features,
                "vision_pos_enc": [p[:, t] for p in vision_pos],
                "backbone_fpn": [f[:, t] for f in backbone_fpn],
            }
            # `_decode` -- not a re-implementation of it. It carries
            # `_suppress_absent_object` (D-043), the D-044-conditional best-IoU
            # pointer gather (D-038), `obj_ptr_proj` and `_blend_object_pointer`
            # (D-039). Copying that tail here would fork four decisions that
            # took a full review round to settle. `SAM2.call` is NOT called:
            # `_decode` is a plain method with no `postprocess`, no memory and
            # no Keras `__call__` bookkeeping of its own.
            outputs = self.sam2._decode(
                encoded=per_frame_encoded,
                features=conditioned,
                inputs=prompt,
                multimask_output=False,
                training=training,
            )
            frame_logits.append(outputs["low_res_logits"])
            frame_scores.append(outputs["object_score_logits"])
            frame_iou.append(
                self._iou_supervision(outputs, gt_masks[:, t:t + 1]))

            self._store(bank, t, features, outputs, training)

        return {
            # Each frame contributes a length-1 mask axis (M == 1), so the
            # concatenation IS the frame axis (D-051).
            SAM2_LOW_RES_LOGITS: ops.concatenate(frame_logits, axis=1),
            SAM2_OBJECT_SCORE_LOGITS: ops.stack(frame_scores, axis=1),
            SAM2_IOU_SUPERVISION: ops.concatenate(frame_iou, axis=1),
        }

    # -----------------------------------------------------------------
    # shapes / config
    # -----------------------------------------------------------------

    def compute_output_shape(
            self, input_shape: Dict[str, Tuple[Optional[int], ...]]
    ) -> Dict[str, Tuple[Optional[int], ...]]:
        """Return the output shapes, derived from stored config.

        :param input_shape: ``{'image': (B, T, S, S, 3)}``.
        :type input_shape: Dict[str, Tuple[Optional[int], ...]]
        :return: One shape per :meth:`call` output key.
        :rtype: Dict[str, Tuple[Optional[int], ...]]
        """
        batch = tuple(input_shape.get(INPUT_IMAGE, (None,)))[0]
        mask_grid = self.sam2.feature_grid * 4
        return {
            SAM2_LOW_RES_LOGITS: (
                batch, self.num_frames, mask_grid, mask_grid),
            SAM2_OBJECT_SCORE_LOGITS: (batch, self.num_frames, 1),
            SAM2_IOU_SUPERVISION: (batch, self.num_frames, 2),
        }

    def get_config(self) -> Dict[str, Any]:
        """Serialize the wrapper, including the whole wrapped SAM 2.

        :return: Configuration dict consumable by :meth:`from_config`.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "sam2": keras.layers.serialize(self.sam2),
            "num_frames": self.num_frames,
            "seed": self.seed,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SAM2TrainingModel":
        """Rebuild a wrapper (and its SAM 2) from :meth:`get_config` output.

        :param config: A dict produced by :meth:`get_config`.
        :type config: Dict[str, Any]
        :return: A new :class:`SAM2TrainingModel`.
        :rtype: SAM2TrainingModel
        """
        config = dict(config)
        config["sam2"] = keras.layers.deserialize(config["sam2"])
        return cls(**config)


def compile_sam2_video_trainer(
        model: SAM2TrainingModel,
        optimizer: Any = "adam",
        mask_weight: float = 1.0,
        object_score_weight: float = 1.0,
        iou_weight: float = 1.0,
        **compile_kwargs: Any,
) -> None:
    """Compile ``model`` with the three losses SAM 2 is trained with.

    This function exists so that the ``loss=`` dict has exactly ONE home. H-5:
    a dict ``y_pred`` requires ``loss=`` keyed to the output NAMES, and nothing
    in Keras checks the two sets against each other -- a key spelled in the
    trainer and again in the pipeline drifts silently. Both read
    :data:`OUTPUT_KEYS` through this function instead.

    :param model: The wrapper to compile.
    :type model: SAM2TrainingModel
    :param optimizer: Any Keras optimizer or its string name.
    :type optimizer: Any
    :param mask_weight: Weight of the gated focal+dice term. The focal:dice
        ratio itself lives inside :class:`SAM2GatedMaskLoss`.
    :type mask_weight: float
    :param object_score_weight: Weight of the object-score BCE. Upstream's
        ``loss_class: 1``.
    :type object_score_weight: float
    :param iou_weight: Weight of the IoU regression term.
    :type iou_weight: float
    :param compile_kwargs: Forwarded to ``keras.Model.compile``. Passing
        ``jit_compile`` here overrides the mandatory ``False`` and will make the
        first ``fit()`` step raise on a GPU; see the module docstring.
    :type compile_kwargs: Any
    """
    # DECISION plan-2026-08-04T044628-4c240b4c/D-052
    # The object-score term is MANDATORY, not optional, and it is stock
    # `BinaryCrossentropy(from_logits=True)` -- upstream's `loss_class` is
    # `sigmoid_focal_loss(..., focal_gamma_obj_score=0.0,
    # focal_alpha_obj_score=-1.0)` at `loss_class: 1`, which IS a plain BCE.
    # Do NOT drop this key to "train the masks first", and do NOT write a
    # bespoke loss class for it (one call site, no behaviour of its own).
    # Dropping it is silent and total: every consumer of `object_score_logits`
    # in this package thresholds it HARD at `> 0` (`_suppress_absent_object`
    # D-043, `_mark_occlusion`, `_blend_object_pointer` at the shipped
    # `soft_no_obj_ptr=False`), so the score head has NO differentiable
    # consumer. A mask-only loss can neither train it nor re-open the mask path
    # it has closed -- `ops.where` passes no gradient through the suppressed
    # branch -- and at random init every score is negative, so the whole mask
    # output is the constant -1024 with a finite, falling, meaningless loss.
    # Likewise `jit_compile=False` is MANDATORY (D-055): `Hiera`'s stem bicubic
    # resize has no XLA GPU kernel and Keras 3.8 defaults `fit()` to
    # `jit_compile='auto'`. See decisions.md D-052 and D-055.
    compile_kwargs.setdefault("jit_compile", False)
    model.compile(
        optimizer=optimizer,
        loss={
            SAM2_LOW_RES_LOGITS: SAM2GatedMaskLoss(),
            SAM2_OBJECT_SCORE_LOGITS: keras.losses.BinaryCrossentropy(
                from_logits=True),
            SAM2_IOU_SUPERVISION: SAMIoULoss(),
        },
        loss_weights={
            SAM2_LOW_RES_LOGITS: float(mask_weight),
            SAM2_OBJECT_SCORE_LOGITS: float(object_score_weight),
            SAM2_IOU_SUPERVISION: float(iou_weight),
        },
        **compile_kwargs,
    )
