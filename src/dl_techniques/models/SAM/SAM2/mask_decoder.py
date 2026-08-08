"""
SAM 2 mask decoder: masks, IoU predictions, an object score and a pointer.
==========================================================================

:class:`SAM2MaskDecoder` is a NEW SIBLING of
:class:`dl_techniques.models.SAM.SAM1.mask_decoder.MaskDecoder`, not an
extension of it. SAM 1's decoder bakes its token layout into positional slices
inside method bodies (``hs[:, 0, :]``, ``hs[:, 1:1 + N, :]``), has no
skip-connection argument in its signature at all, and returns a 2-tuple; none
of the SAM 2 deltas below can be expressed as a defaulted ``__init__`` kwarg.
SAM 1 is imported from, never edited, by this file.

Based on:
---------
- Ravi, N. et al. (2024). "SAM 2: Segment Anything in Images and Videos."

Key Features:
------------
- A token block of ``concat([obj_score_token, iou_token, mask_tokens])``.
- Two high-resolution skip connections from the image encoder's finer levels.
- A stability score per mask, and a per-batch-element fallback that uses it.

Architecture Overview:
---------------------
1. Tokens and image embedding go through the two-way transformer.
2. -> two transposed-convolution upscaling steps, each taking one skip.
3. -> per-mask hypernetwork MLPs dotted against the upscaled embedding.
4. -> the IoU head, the object-score head, and the stability-based selection.

Usage Examples:
--------------
```python
from dl_techniques.models.SAM.SAM2.mask_decoder import SAM2MaskDecoder
decoder = SAM2MaskDecoder(transformer_dim=256)
low_res_logits, iou_predictions, object_score_logits, object_pointer = (
    decoder(image_embeddings, image_pe, sparse, dense, multimask_output=True))
```

Measured caveats:
----------------
Four mechanisms are SILENT when ported wrong -- the layer builds,
forward-passes, trains and serializes either way. All four are guarded
behaviourally in ``tests/test_models/test_sam2/test_mask_decoder.py``:

- **The object-score token is PREPENDED**, so every subsequent token index
  shifts by ``s = 1``. The block is
  ``concat([obj_score_token, iou_token, mask_tokens])``; the IoU token is
  ``hs[:, s, :]`` and the mask tokens are ``hs[:, s + 1 : s + 1 + N, :]``. The
  object score is read from ``hs[:, 0, :]`` -- the obj-score token's OWN
  transformer output, not a separate branch off the IoU token. Reading index 1
  yields the same shapes and a plausible score.
- **The high-resolution skips are ADDED**, before the norm/activation, never
  concatenated: ``act1(ln1(dc1(src) + feat_s1))`` then
  ``act2(dc2(upscaled) + feat_s0)``. A coherent concat port keeps every output
  SHAPE and changes only the DECLARED widths, which is why the guard asserts
  the width and not merely that the value moved.
- **The stability score is a self-consistency measure**, not an IoU against
  ground truth: ``area_i = sum(logits > +delta)``,
  ``area_u = sum(logits > -delta)``. Swapping the two deltas produces a
  perfectly finite score in ``[1, inf)`` and never raises.
- **The unstable-case fallback is PER BATCH ELEMENT.** A single global
  ``argmax`` over the batch is shape-identical and is invisible at batch 1.
"""

import keras
from keras import layers, ops
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.norms import create_normalization_layer

# Reuse, do not re-implement: SAM 1's transformer and its MLP-head builder are
# imported UNCHANGED (a second instance of the former, a plain function call for
# the latter). `models/SAM/SAM1/__init__.py` does not re-export either name, so both
# imports must name the submodule directly -- exactly as `train_sam.py` does.
from dl_techniques.models.SAM.SAM1.transformer import TwoWayTransformer
from dl_techniques.models.SAM.SAM1.mask_decoder import _build_mlp_head

# ---------------------------------------------------------------------

#: Depth of each mask-token hypernetwork MLP. Fixed by the reference
#: implementation (``MLP(dim, dim, dim // 8, 3)``) and deliberately independent
#: of ``iou_head_depth``, which IS a constructor parameter. See D-035.
_HYPERNETWORK_MLP_DEPTH = 3

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class SAM2MaskDecoder(keras.layers.Layer):
    """Predict masks, IoU scores, an object score and an object pointer.

    The SAM 2 mask decoder. Relative to SAM 1 it adds an object-score token and
    head, two additive high-resolution skip connections into the upscaling
    pathway, a stability-based dynamic choice between the single-mask and
    multimask tokens, and an object-pointer output feeding the memory bank.

    :param transformer_dim: Embedding width of the transformer and of the input
        embeddings.
    :type transformer_dim: int
    :param transformer: The two-way transformer instance. Construct a SECOND
        :class:`~dl_techniques.models.SAM.SAM1.transformer.TwoWayTransformer` at
        ``(depth=2, embedding_dim=256, num_heads=8, mlp_dim=2048)``; do not
        share SAM 1's instance and do not subclass it.
    :type transformer: TwoWayTransformer
    :param num_multimask_outputs: Number of multimask tokens. The total token
        count is ``num_multimask_outputs + 1`` (the extra one is the
        single-mask token at index 0). Defaults to ``3``.
    :type num_multimask_outputs: int
    :param iou_head_depth: Total number of ``Dense`` layers in each MLP head.
        Defaults to ``3``.
    :type iou_head_depth: int
    :param iou_head_hidden_dim: Hidden width of the IoU head. Defaults to
        ``256``.
    :type iou_head_hidden_dim: int
    :param use_high_res_features: Whether the two high-resolution skips are
        consumed. When ``True``, ``call`` REQUIRES ``high_res_features``.
        Defaults to ``False``.
    :type use_high_res_features: bool
    :param iou_prediction_use_sigmoid: Whether to squash the IoU head output
        through a sigmoid. Defaults to ``False``.
    :type iou_prediction_use_sigmoid: bool
    :param pred_obj_scores: Whether to prepend an object-score token and read a
        learned object score from it. When ``False`` the object score is the
        constant ``10.0`` (``sigmoid(10) ~ 1``, i.e. "always assume an object").
        Defaults to ``False``.
    :type pred_obj_scores: bool
    :param pred_obj_scores_mlp: Whether the object-score head is a 3-layer MLP
        rather than a single linear layer. Defaults to ``False``.
    :type pred_obj_scores_mlp: bool
    :param use_multimask_token_for_obj_ptr: Whether the object pointer is taken
        from the multimask tokens when ``multimask_output`` is set. Defaults to
        ``False``, which sources it from the single-mask token ALWAYS.
    :type use_multimask_token_for_obj_ptr: bool
    :param dynamic_multimask_via_stability: Whether, at inference with
        ``multimask_output=False``, an unstable single-mask prediction falls
        back to the best multimask token. Defaults to ``False``.
    :type dynamic_multimask_via_stability: bool
    :param dynamic_multimask_stability_delta: The logit offset defining the two
        thresholded areas. Defaults to ``0.05``.
    :type dynamic_multimask_stability_delta: float
    :param dynamic_multimask_stability_thresh: Stability at or above which the
        single-mask prediction is kept. Defaults to ``0.98``.
    :type dynamic_multimask_stability_thresh: float
    :param normalization_type: Normalization used inside the upscaling
        pathway. Defaults to ``'layer_norm'``.
    :type normalization_type: str
    :param activation: Activation inside the upscaling pathway ONLY. Defaults
        to ``'gelu'``.
    :type activation: str
    :param mlp_activation: Activation on every non-final layer of the
        hypernetwork MLPs, the IoU head and the object-score MLP. Defaults to
        ``'relu'``. This is a SEPARATE knob from ``activation`` because the
        reference implementation makes the two halves differ; see
        ``models/SAM/SAM1/mask_decoder.py``'s D-024 anchor.
    :type mlp_activation: str

    :raises ValueError: if any positive-valued argument is non-positive, or if
        ``transformer_dim`` is not divisible by 8.

    Example::

        from dl_techniques.models.SAM.SAM1.transformer import TwoWayTransformer
        from dl_techniques.models.SAM.SAM2.mask_decoder import SAM2MaskDecoder

        decoder = SAM2MaskDecoder(
            transformer_dim=256,
            transformer=TwoWayTransformer(
                depth=2, embedding_dim=256, num_heads=8, mlp_dim=2048
            ),
            use_high_res_features=True,
            pred_obj_scores=True,
        )
        logits, iou, obj_score, obj_ptr = decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=True,
            high_res_features=[feat_4x, feat_2x],
        )
    """

    def __init__(
        self,
        *,
        transformer_dim: int = 256,
        transformer: TwoWayTransformer,
        num_multimask_outputs: int = 3,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_high_res_features: bool = False,
        iou_prediction_use_sigmoid: bool = False,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        use_multimask_token_for_obj_ptr: bool = False,
        dynamic_multimask_via_stability: bool = False,
        dynamic_multimask_stability_delta: float = 0.05,
        dynamic_multimask_stability_thresh: float = 0.98,
        normalization_type: Literal['layer_norm', 'rms_norm', 'batch_norm'] = 'layer_norm',
        activation: str = 'gelu',
        mlp_activation: str = 'relu',
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if transformer_dim <= 0:
            raise ValueError(
                f"transformer_dim must be positive, got {transformer_dim}"
            )
        if transformer_dim % 8 != 0:
            raise ValueError(
                f"transformer_dim must be divisible by 8, got {transformer_dim}: "
                f"the upscaling pathway emits transformer_dim // 8 channels and "
                f"the hypernetwork MLPs must match that width exactly, so a "
                f"floored division is a silent width mismatch."
            )
        if num_multimask_outputs <= 0:
            raise ValueError(
                f"num_multimask_outputs must be positive, got "
                f"{num_multimask_outputs}"
            )
        if iou_head_depth <= 0:
            raise ValueError(
                f"iou_head_depth must be positive, got {iou_head_depth}"
            )
        if iou_head_hidden_dim <= 0:
            raise ValueError(
                f"iou_head_hidden_dim must be positive, got {iou_head_hidden_dim}"
            )
        if not 0.0 < dynamic_multimask_stability_thresh <= 1.0:
            raise ValueError(
                f"dynamic_multimask_stability_thresh must lie in (0, 1], got "
                f"{dynamic_multimask_stability_thresh}. The score is a ratio of "
                f"two thresholded areas and can never exceed 1, so a larger "
                f"threshold silently disables the single-mask branch entirely."
            )
        if dynamic_multimask_stability_delta <= 0.0:
            raise ValueError(
                f"dynamic_multimask_stability_delta must be positive, got "
                f"{dynamic_multimask_stability_delta}: at delta == 0 the two "
                f"areas coincide and the score is the constant 1.0."
            )

        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs
        self.iou_head_depth = iou_head_depth
        self.iou_head_hidden_dim = iou_head_hidden_dim
        self.use_high_res_features = use_high_res_features
        self.iou_prediction_use_sigmoid = iou_prediction_use_sigmoid
        self.pred_obj_scores = pred_obj_scores
        self.pred_obj_scores_mlp = pred_obj_scores_mlp
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr
        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh
        self.normalization_type = normalization_type
        self.activation = activation
        self.mlp_activation = mlp_activation

        self.num_mask_tokens = num_multimask_outputs + 1

        # DECISION plan-2026-08-04T044628-4c240b4c/D-021
        # `token_offset` is `s` from the reference: the number of tokens sitting
        # BEFORE the IoU token. Do NOT hardcode 0 "because that is what SAM 1
        # does", and do NOT hardcode 1 "because SAM 2 always predicts object
        # scores". Both are silently wrong for the other configuration: every
        # downstream slice (`hs[:, s, :]`, `hs[:, s + 1 : s + 1 + N, :]`) keeps
        # its shape under an off-by-one, so the decoder reads the IoU head off a
        # mask token and every test that only checks shapes stays green. See
        # decisions.md D-021.
        self.token_offset = 1 if pred_obj_scores else 0

        # CREATE all sub-layers in __init__, UNCONDITIONALLY.
        #
        # `obj_score_token`, `pred_obj_score_head`, `conv_s0` and `conv_s1` are
        # created even when their feature flag is off. Creating a sub-layer
        # inside an `if` makes the set of weights depend on a config value that
        # `from_config` restores AFTER `__init__` has already run in some load
        # paths, which is the documented silent-weight-drop failure mode for
        # this repo. The cost is a few unused (but built and saved) weights in
        # the flags-off configuration; the shipped SAM 2 configuration turns
        # every one of these flags ON.
        self.iou_token = layers.Embedding(1, transformer_dim, name="iou_token")
        self.mask_tokens = layers.Embedding(
            self.num_mask_tokens, transformer_dim, name="mask_tokens"
        )
        self.obj_score_token = layers.Embedding(
            1, transformer_dim, name="obj_score_token"
        )

        # The upscaling pathway is held as FIVE NAMED sub-layers rather than one
        # `keras.Sequential`, because the high-resolution skips are injected
        # BETWEEN them (`dc1 -> +feat_s1 -> ln1 -> act1 -> dc2 -> +feat_s0 ->
        # act2`). A Sequential cannot express that, and destructuring one at
        # call time is exactly the reference's own workaround.
        self.dc1 = layers.Conv2DTranspose(
            transformer_dim // 4, kernel_size=2, strides=2, name="upsample_conv1"
        )
        self.ln1 = create_normalization_layer(
            normalization_type, name="upsample_norm1"
        )
        self.act1 = layers.Activation(activation, name="upsample_act1")
        self.dc2 = layers.Conv2DTranspose(
            transformer_dim // 8, kernel_size=2, strides=2, name="upsample_conv2"
        )
        self.act2 = layers.Activation(activation, name="upsample_act2")

        # Lateral 1x1 projections for the two high-resolution skips. The neck
        # emits every FPN level at `transformer_dim` channels, so these map
        # `transformer_dim -> transformer_dim // 8` (the 4x level, added after
        # `dc2`) and `transformer_dim -> transformer_dim // 4` (the 2x level,
        # added after `dc1`).
        #
        # DECISION plan-2026-08-04T044628-4c240b4c/D-022
        # These convolutions are APPLIED HERE, inside `predict_masks`. The
        # reference declares them on the decoder but applies them from the
        # top-level model, which makes the decoder silently dependent on its
        # caller having done half of its own forward pass -- a reach-in that
        # step 8 would have had to reproduce, and that makes the decoder
        # untestable in isolation. Do NOT "restore fidelity" by moving them out;
        # the weights, their shapes and their names are identical either way.
        # See decisions.md D-022.
        self.conv_s0 = layers.Conv2D(
            transformer_dim // 8, kernel_size=1, name="conv_s0"
        )
        self.conv_s1 = layers.Conv2D(
            transformer_dim // 4, kernel_size=1, name="conv_s1"
        )

        # DECISION plan-2026-08-04T044628-4c240b4c/D-035
        # The hypernetwork MLPs are FIXED at 3 layers and are deliberately NOT
        # tied to `iou_head_depth`. Do NOT "deduplicate" the two by passing
        # `self.iou_head_depth` here: the reference hardcodes
        # `MLP(dim, dim, dim // 8, 3)` for the hypernetworks while exposing the
        # IoU head's depth as a parameter. The two agree at the default
        # `iou_head_depth=3`, so the coupling is invisible at every shipped
        # configuration and silently restructures the mask heads at any other.
        # See decisions.md D-035.
        self.output_hypernetworks_mlps: List[keras.Sequential] = []
        for i in range(self.num_mask_tokens):
            self.output_hypernetworks_mlps.append(
                _build_mlp_head(
                    num_layers=_HYPERNETWORK_MLP_DEPTH,
                    hidden_dim=transformer_dim,
                    output_dim=transformer_dim // 8,
                    activation=mlp_activation,
                    dense_name_template=f"hyper_dense{{n}}_{i}",
                    name=f"hypernetwork_mlp_{i}",
                )
            )

        self.iou_prediction_head = _build_mlp_head(
            num_layers=self.iou_head_depth,
            hidden_dim=self.iou_head_hidden_dim,
            output_dim=self.num_mask_tokens,
            activation=mlp_activation,
            dense_name_template="iou_dense{n}",
            name="iou_prediction_head",
        )

        self.pred_obj_score_head = _build_mlp_head(
            num_layers=3 if pred_obj_scores_mlp else 1,
            hidden_dim=transformer_dim,
            output_dim=1,
            activation=mlp_activation,
            dense_name_template="obj_score_dense{n}",
            name="pred_obj_score_head",
        )

    def build(self, input_shape: Optional[Tuple[Optional[int], ...]] = None) -> None:
        """Explicitly build every sub-layer.

        :param input_shape: Unused; the decoder's real input shapes arrive as
            ``call`` arguments.
        :type input_shape: Optional[Tuple[Optional[int], ...]]
        """
        self.iou_token.build((None,))
        self.mask_tokens.build((None,))
        self.obj_score_token.build((None,))

        # The transformer builds its own sub-layers lazily from the actual
        # query/key shapes on first call, matching SAM 1's decoder.
        self.dc1.build((None, None, None, self.transformer_dim))
        self.ln1.build((None, None, None, self.transformer_dim // 4))
        self.dc2.build((None, None, None, self.transformer_dim // 4))

        self.conv_s0.build((None, None, None, self.transformer_dim))
        self.conv_s1.build((None, None, None, self.transformer_dim))

        for mlp in self.output_hypernetworks_mlps:
            mlp.build((None, self.transformer_dim))
        self.iou_prediction_head.build((None, self.transformer_dim))
        self.pred_obj_score_head.build((None, self.transformer_dim))

        super().build(input_shape)

    # -----------------------------------------------------------------
    # forward
    # -----------------------------------------------------------------

    def call(
        self,
        image_embeddings: keras.KerasTensor,
        image_pe: keras.KerasTensor,
        sparse_prompt_embeddings: keras.KerasTensor,
        dense_prompt_embeddings: keras.KerasTensor,
        multimask_output: bool,
        high_res_features: Optional[Sequence[keras.KerasTensor]] = None,
        training: Optional[bool] = None,
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]:
        """Run the decoder and select the output masks.

        :param image_embeddings: ``(B, H, W, transformer_dim)`` image features.
        :type image_embeddings: keras.KerasTensor
        :param image_pe: ``(B, H, W, transformer_dim)`` positional encoding.
        :type image_pe: keras.KerasTensor
        :param sparse_prompt_embeddings: ``(B or 1, N, transformer_dim)``.
        :type sparse_prompt_embeddings: keras.KerasTensor
        :param dense_prompt_embeddings: ``(B, H, W, transformer_dim)``.
        :type dense_prompt_embeddings: keras.KerasTensor
        :param multimask_output: Whether to return the multimask tokens.
        :type multimask_output: bool
        :param high_res_features: ``[feat_4x, feat_2x]``, each
            ``(B, h, w, transformer_dim)`` at 4x and 2x the embedding grid.
            Required iff ``use_high_res_features``.
        :type high_res_features: Optional[Sequence[keras.KerasTensor]]
        :param training: Keras training flag.
        :type training: Optional[bool]
        :return: ``(low_res_logits, iou_predictions, object_score_logits,
            object_pointer)``.
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]
        """
        masks, iou_pred, mask_tokens_out, object_score_logits = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            high_res_features=high_res_features,
            training=training,
        )

        if multimask_output:
            masks = masks[:, 1:, :, :]
            iou_pred = iou_pred[:, 1:]
        elif self.dynamic_multimask_via_stability and not bool(training):
            masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
        else:
            masks = masks[:, 0:1, :, :]
            iou_pred = iou_pred[:, 0:1]

        # DECISION plan-2026-08-04T044628-4c240b4c/D-023
        # The object pointer comes from mask token 0 -- the SINGLE-mask token --
        # unless BOTH `multimask_output` and `use_multimask_token_for_obj_ptr`
        # hold. Do NOT "simplify" this to follow `multimask_output` alone so it
        # agrees with the mask selection above. Training always runs the
        # single-mask token (after the first click multimask degenerates to
        # single), so a pointer sourced from a multimask token at test time is
        # fed to a memory bank that never saw one during training. Shapes and
        # dtypes are unaffected. See decisions.md D-023.
        if multimask_output and self.use_multimask_token_for_obj_ptr:
            object_pointer = mask_tokens_out[:, 1:, :]
        else:
            object_pointer = mask_tokens_out[:, 0:1, :]

        return masks, iou_pred, object_score_logits, object_pointer

    def predict_masks(
        self,
        image_embeddings: keras.KerasTensor,
        image_pe: keras.KerasTensor,
        sparse_prompt_embeddings: keras.KerasTensor,
        dense_prompt_embeddings: keras.KerasTensor,
        high_res_features: Optional[Sequence[keras.KerasTensor]] = None,
        training: Optional[bool] = None,
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]:
        """Produce ALL mask tokens' logits, before any output selection.

        :param image_embeddings: ``(B, H, W, transformer_dim)``.
        :type image_embeddings: keras.KerasTensor
        :param image_pe: ``(B, H, W, transformer_dim)``.
        :type image_pe: keras.KerasTensor
        :param sparse_prompt_embeddings: ``(B or 1, N, transformer_dim)``.
        :type sparse_prompt_embeddings: keras.KerasTensor
        :param dense_prompt_embeddings: ``(B, H, W, transformer_dim)``.
        :type dense_prompt_embeddings: keras.KerasTensor
        :param high_res_features: ``[feat_4x, feat_2x]`` or ``None``.
        :type high_res_features: Optional[Sequence[keras.KerasTensor]]
        :param training: Keras training flag.
        :type training: Optional[bool]
        :return: ``(masks, iou_predictions, mask_tokens_out,
            object_score_logits)`` with ``masks`` of shape
            ``(B, num_mask_tokens, H * 4, W * 4)``.
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]
        :raises ValueError: if ``use_high_res_features`` and
            ``high_res_features`` disagree, or if the sparse prompt batch is
            neither ``1`` nor the image batch size.
        """
        if self.use_high_res_features and high_res_features is None:
            raise ValueError(
                "SAM2MaskDecoder was built with use_high_res_features=True but "
                "call() received high_res_features=None. Without the skips the "
                "decoder still produces correctly shaped masks from a strictly "
                "coarser feature stream, so this would be a silent quality "
                "regression rather than an error. Pass [feat_4x, feat_2x]."
            )
        if high_res_features is not None and not self.use_high_res_features:
            raise ValueError(
                "SAM2MaskDecoder received high_res_features but was built with "
                "use_high_res_features=False, so they would be silently "
                "discarded. Set use_high_res_features=True."
            )
        if high_res_features is not None and len(high_res_features) != 2:
            raise ValueError(
                f"high_res_features must be exactly [feat_4x, feat_2x], got "
                f"{len(high_res_features)} tensors."
            )

        # Token block: [obj_score_token?, iou_token, mask_tokens].
        token_parts = []
        if self.pred_obj_scores:
            token_parts.append(self.obj_score_token.weights[0])
        token_parts.append(self.iou_token.weights[0])
        token_parts.append(self.mask_tokens.weights[0])
        output_tokens = ops.concatenate(token_parts, axis=0)

        output_tokens = ops.expand_dims(output_tokens, 0)
        batch_size = ops.shape(image_embeddings)[0]
        output_tokens = ops.broadcast_to(
            output_tokens,
            (batch_size, ops.shape(output_tokens)[1], ops.shape(output_tokens)[2]),
        )

        # Only two sparse batch sizes are meaningful: 1 (one prompt set shared
        # by the image batch) and exactly B. Anything else either floors the
        # tile factor to 0 or interleaves prompts against the wrong images with
        # no error at all -- SAM 1 measured both. The check reads STATIC shapes,
        # so under tracing with an unknown batch it is skipped rather than
        # traced.
        static_batch = image_embeddings.shape[0]
        static_sparse = sparse_prompt_embeddings.shape[0]
        if (
            static_batch is not None
            and static_sparse is not None
            and static_sparse not in (1, static_batch)
        ):
            raise ValueError(
                f"SAM2MaskDecoder cannot tile {static_sparse} sparse prompt "
                f"rows onto an image batch of {static_batch}: sparse_batch "
                f"must be 1 or exactly batch_size={static_batch}."
            )
        sparse_batch = ops.shape(sparse_prompt_embeddings)[0]
        sparse_prompt_embeddings = ops.tile(
            sparse_prompt_embeddings, [batch_size // sparse_batch, 1, 1]
        )

        tokens = ops.concatenate([output_tokens, sparse_prompt_embeddings], axis=1)

        src = image_embeddings + dense_prompt_embeddings
        pos_src = image_pe
        b, h, w, c = ops.shape(src)

        hs, src_out = self.transformer(src, pos_src, tokens, training=training)

        s = self.token_offset
        iou_token_out = hs[:, s, :]
        mask_tokens_out = hs[:, s + 1 : s + 1 + self.num_mask_tokens, :]

        src_out = ops.reshape(src_out, (b, h, w, c))

        # Additive high-resolution fusion, BEFORE the norm/activation.
        if self.use_high_res_features:
            feat_s0 = self.conv_s0(high_res_features[0], training=training)
            feat_s1 = self.conv_s1(high_res_features[1], training=training)
            upscaled = self.dc1(src_out, training=training) + feat_s1
            upscaled = self.act1(self.ln1(upscaled, training=training))
            upscaled = self.dc2(upscaled, training=training) + feat_s0
            upscaled = self.act2(upscaled)
        else:
            upscaled = self.dc1(src_out, training=training)
            upscaled = self.act1(self.ln1(upscaled, training=training))
            upscaled = self.act2(self.dc2(upscaled, training=training))

        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](
                    mask_tokens_out[:, i, :], training=training
                )
            )
        hyper_in = ops.stack(hyper_in_list, axis=1)

        b_up, h_up, w_up, c_up = ops.shape(upscaled)
        upscaled_flat = ops.reshape(upscaled, (b_up, h_up * w_up, c_up))
        masks = ops.matmul(hyper_in, ops.transpose(upscaled_flat, (0, 2, 1)))
        masks = ops.reshape(masks, (b_up, self.num_mask_tokens, h_up, w_up))

        iou_pred = self.iou_prediction_head(iou_token_out, training=training)
        if self.iou_prediction_use_sigmoid:
            iou_pred = ops.sigmoid(iou_pred)

        if self.pred_obj_scores:
            # Index 0 is the obj-score token's OWN transformer output.
            object_score_logits = self.pred_obj_score_head(
                hs[:, 0, :], training=training
            )
        else:
            object_score_logits = 10.0 * ops.ones(
                (batch_size, 1), dtype=masks.dtype
            )

        return masks, iou_pred, mask_tokens_out, object_score_logits

    # -----------------------------------------------------------------
    # stability
    # -----------------------------------------------------------------

    def _get_stability_scores(
        self, mask_logits: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Compute the threshold-shift self-consistency score of each mask.

        The score is the IoU between the mask thresholded at ``+delta`` and the
        mask thresholded at ``-delta``. It measures robustness to a threshold
        shift; it is NOT an IoU against ground truth and uses no labels.

        The two areas are COUNTS, and are accumulated in **float32 regardless
        of the input dtype** -- see the decision anchor below.

        :param mask_logits: ``(B, M, H, W)`` mask logits.
        :type mask_logits: keras.KerasTensor
        :return: ``(B, M)`` stability scores in ``[0, 1]``, always float32.
        :rtype: keras.KerasTensor
        """
        # DECISION plan-2026-08-04T044628-4c240b4c/D-034
        # The two areas are summed in float32, NEVER in `mask_logits.dtype`.
        # Do NOT "simplify" this back to the input dtype: at the shipped
        # `image_size=1024` this head sees 256 x 256 = 65,536 logits per mask,
        # and float16's largest finite value is 65,504. MEASURED under
        # `mixed_float16` at that exact size: an all-positive mask gives
        # `stability = [nan nan nan nan]`, `NaN >= 0.98` evaluates False, and a
        # maximally-confident single mask is therefore SILENTLY replaced by a
        # multimask token on the default `training=None` path -- a behaviour
        # INVERSION, not merely a NaN. A toy grid cannot reproduce it; the arm
        # in `test_mask_decoder.py` that guards this runs at the shipped
        # 256x256. See decisions.md D-034.
        delta = self.dynamic_multimask_stability_delta
        shape = ops.shape(mask_logits)
        flat = ops.reshape(mask_logits, (shape[0], shape[1], -1))
        area_i = ops.sum(ops.cast(flat > delta, "float32"), axis=-1)
        area_u = ops.sum(ops.cast(flat > -delta, "float32"), axis=-1)
        # `area_u == 0` means the mask is empty even at the permissive
        # threshold: it is trivially self-consistent, so the score is 1.0. The
        # division is made safe FIRST -- a `where` alone still evaluates the
        # NaN branch and poisons the gradient.
        safe_u = ops.where(area_u > 0, area_u, ops.ones_like(area_u))
        return ops.where(area_u > 0, area_i / safe_u, ops.ones_like(area_i))

    def _dynamic_multimask_via_stability(
        self,
        all_mask_logits: keras.KerasTensor,
        all_iou_scores: keras.KerasTensor,
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Fall back to the best multimask token when token 0 is unstable.

        :param all_mask_logits: ``(B, num_mask_tokens, H, W)``.
        :type all_mask_logits: keras.KerasTensor
        :param all_iou_scores: ``(B, num_mask_tokens)``.
        :type all_iou_scores: keras.KerasTensor
        :return: ``(masks, iou_predictions)``, each with a mask axis of size 1.
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor]
        """
        multimask_logits = all_mask_logits[:, 1:, :, :]
        multimask_iou = all_iou_scores[:, 1:]

        # DECISION plan-2026-08-04T044628-4c240b4c/D-024
        # The argmax is PER BATCH ELEMENT and the gather is
        # `take_along_axis(..., axis=1)`. Do NOT replace it with a single
        # `ops.argmax(multimask_iou)` over the whole tensor followed by a
        # slice: the output shape is identical, and at batch size 1 the two are
        # numerically indistinguishable, so a batch-1 fixture proves nothing.
        # See decisions.md D-024.
        best_idx = ops.argmax(multimask_iou, axis=-1)
        idx_masks = ops.reshape(best_idx, (-1, 1, 1, 1))
        best_multimask_logits = ops.take_along_axis(
            multimask_logits, idx_masks, axis=1
        )
        best_multimask_iou = ops.take_along_axis(
            multimask_iou, ops.reshape(best_idx, (-1, 1)), axis=1
        )

        singlemask_logits = all_mask_logits[:, 0:1, :, :]
        singlemask_iou = all_iou_scores[:, 0:1]

        stability = self._get_stability_scores(singlemask_logits)
        is_stable = stability >= self.dynamic_multimask_stability_thresh

        masks_out = ops.where(
            ops.reshape(is_stable, (-1, 1, 1, 1)),
            singlemask_logits,
            best_multimask_logits,
        )
        iou_out = ops.where(is_stable, singlemask_iou, best_multimask_iou)
        return masks_out, iou_out

    # -----------------------------------------------------------------
    # shapes / config
    # -----------------------------------------------------------------

    def compute_output_shape(
        self, input_shape: Optional[Tuple[Optional[int], ...]] = None
    ) -> Tuple[Tuple[Optional[int], ...], ...]:
        """Return the four output shapes.

        :param input_shape: Unused.
        :type input_shape: Optional[Tuple[Optional[int], ...]]
        :return: ``(mask_shape, iou_shape, object_score_shape,
            object_pointer_shape)``.
        :rtype: Tuple[Tuple[Optional[int], ...], ...]
        """
        return (
            (None, None, None, None),
            (None, None),
            (None, 1),
            (None, None, self.transformer_dim),
        )

    def get_config(self) -> Dict[str, Any]:
        """Return the full constructor configuration.

        :return: Serializable configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "transformer_dim": self.transformer_dim,
            "transformer": keras.layers.serialize(self.transformer),
            "num_multimask_outputs": self.num_multimask_outputs,
            "iou_head_depth": self.iou_head_depth,
            "iou_head_hidden_dim": self.iou_head_hidden_dim,
            "use_high_res_features": self.use_high_res_features,
            "iou_prediction_use_sigmoid": self.iou_prediction_use_sigmoid,
            "pred_obj_scores": self.pred_obj_scores,
            "pred_obj_scores_mlp": self.pred_obj_scores_mlp,
            "use_multimask_token_for_obj_ptr": self.use_multimask_token_for_obj_ptr,
            "dynamic_multimask_via_stability": self.dynamic_multimask_via_stability,
            "dynamic_multimask_stability_delta": self.dynamic_multimask_stability_delta,
            "dynamic_multimask_stability_thresh": self.dynamic_multimask_stability_thresh,
            "normalization_type": self.normalization_type,
            "activation": self.activation,
            "mlp_activation": self.mlp_activation,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SAM2MaskDecoder":
        """Rebuild from :meth:`get_config`, deserializing the transformer.

        :param config: Configuration dictionary.
        :type config: Dict[str, Any]
        :return: A new decoder.
        :rtype: SAM2MaskDecoder
        """
        config["transformer"] = keras.layers.deserialize(config.pop("transformer"))
        return cls(**config)
