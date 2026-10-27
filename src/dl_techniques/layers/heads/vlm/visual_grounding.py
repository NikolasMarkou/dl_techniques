"""``VisualGroundingHead``: scores image regions against a text query and regresses a box."""

import keras
from typing import Dict, List, Optional, Union, Tuple, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..masking import fill_masked
from ...sequence_pooling import SequencePooling
from ..validation import static_rank
from dl_techniques.utils.keras_registration import register_dl_technique
from .utils import _check_task_config_fields, _pool_sequence
from .base import BaseVLMHead


@register_dl_technique("dl_techniques.layers.heads.vlm.visual_grounding")
class VisualGroundingHead(BaseVLMHead):
    """
    Scores image regions against a text query and boxes the best one.

    This head fuses each visual region with the pooled text query, scores
    every region, then regresses a bounding box from the single top-scoring
    region. It inherits from :class:`BaseVLMHead` and uses its ``fusion``
    layer, defaulting the strategy to ``"gated"``.

    It runs no post-fusion stack. ``call`` goes straight from ``fusion`` to
    ``confidence_scorer``, so the head owns no post-fusion norm or FFN and
    takes none of their arguments: ``normalization_type``,
    ``use_post_fusion_ffn``, ``ffn_type`` and ``ffn_expansion_factor`` raise
    ``ValueError`` (unrecognized keyword) here. They belong to
    :class:`ImageTextMatchingHead`, the one head that runs them.

    ``vision_features`` must be rank 3. The regions are whatever the caller
    put on axis 1; this head does not propose them.

    Both inputs must also be ``hidden_dim`` wide. ``D_txt`` and ``D_vis`` are
    not independent, despite the separate symbols on the diagram edges below.
    With ``hidden_size=64``, a 32-wide text stream raises a ``ValueError``
    naming this head and the shapes, around the fusion layer's own "Modality
    1 dimension 32 doesn't match expected dim 64". Project both streams to
    one width before this head.

    The two outputs are treated differently, and the diagram says which is
    which. ``confidence_scorer`` is a linear ``Dense(1)``, so ``'confidence'``
    holds one pre-sigmoid LOGIT per region: it is not confined to ``[0, 1]``.
    Train it with ``BinaryCrossentropy(from_logits=True)``, or apply
    ``keras.ops.sigmoid`` yourself for a score. The region pick is
    unaffected, because the sigmoid is monotonic and ``argmax`` of the
    logits is ``argmax`` of their sigmoid. The box is a range map, not a
    class score: ``bbox_regressor`` is linear, and ``call`` applies
    ``keras.ops.sigmoid`` to its output so ``'bbox'`` lies in ``(0, 1)``.

    **Architecture Overview:**

    .. code-block:: text

        text_features (B, S, D_txt)
                             │
                             ▼
                  ┌─────────────────────┐
                  │ mean over axis 1    │
                  └──────────┬──────────┘
                             ▼ (B, D_txt)
                  ┌─────────────────────┐
                  │ tile to N_regions   │
                  └──────────┬──────────┘
                             ▼
        vision_features ─────┤ (B, N_regions, D_vis)
                             ▼
                  ┌─────────────────────┐
                  │ fusion, gated       │
                  └──────────┬──────────┘
                             ▼ (B, N_regions, hidden_dim)
                  ┌─────────────────────┐
                  │ confidence_scorer   │
                  │ Dense 1, linear     │
                  └──────────┬──────────┘
                             ▼ squeeze
                    ┌────────┴────────┐
                    ▼                 ▼
             'confidence'      argmax over regions
            (B, N_regions)     take_along_axis
            region logits
                                  │
                                  ▼ (B, hidden_dim)
                          ┌───────┴─────────┐
                          ▼                 ▼
               ┌─────────────────────┐  'grounded_features'
               │ bbox_regressor      │
               │ Dense 4, linear     │
               └──────────┬──────────┘
                          ▼
               ┌─────────────────────┐
               │ keras.ops.sigmoid in call │
               └──────────┬──────────┘
                          ▼
                    'bbox' (B, 4)

    The region is picked with ``keras.ops.argmax`` and ``keras.ops.take_along_axis``. See
    the note in ``call`` about why NumPy fancy indexing cannot be used.

    Two optional keep masks (``(batch, length)``, 1 = real): ``'text_mask'``
    restricts the query mean to the real text tokens, and ``'vision_mask'``
    keeps a padded region out of the pick, so ``'bbox'`` and
    ``'grounded_features'`` come from a real region. ``'confidence'`` still
    holds one logit for every region, padded ones included; leave those out
    of the loss.

    Input shape:
        ``{'vision_features': (batch, num_regions, hidden_dim),
        'text_features': (batch, seq_len, hidden_dim)}``, plus the optional
        masks. A rank-2 ``text_features`` is accepted and used as the query
        directly; a ``'text_mask'`` for it is refused.

    Output shape:
        ``{'bbox': (batch, 4), 'confidence': (batch, num_regions),
        'grounded_features': (batch, hidden_dim)}``. ``'bbox'`` is in
        ``(0, 1)``; ``'confidence'`` is region logits.

    :param kwargs: Arguments for :class:`BaseVLMHead` (``task_config``,
        ``fusion_strategy``, ``fusion_config`` and the Keras base arguments).
        ``fusion_strategy`` defaults to ``"gated"`` here rather than to the
        base class value.

    :ivar bbox_regressor: Linear ``Dense`` producing the four box values;
        ``call`` applies the sigmoid.
    :vartype bbox_regressor: keras.layers.Dense
    :ivar confidence_scorer: Linear ``Dense`` producing one logit per region.
    :vartype confidence_scorer: keras.layers.Dense
    :ivar text_pooling: Weightless ``SequencePooling('mean')`` for the query.
    :vartype text_pooling: SequencePooling
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Default the fusion strategy, then create the task layers.

        :param kwargs: Arguments for :class:`BaseVLMHead`.
        :return: None.
        :rtype: None
        :raises ValueError: If ``task_config`` sets ``vocab_size`` or
            ``num_classes``, which this head does not read.
        """
        if "task_config" in kwargs:
            _check_task_config_fields(kwargs["task_config"], "VisualGroundingHead")
        # A gated strategy scores per-region interactions, which is what this
        # head needs; the caller can still override it.
        kwargs.setdefault("fusion_strategy", "gated")
        super().__init__(**kwargs)

        self.text_pooling = SequencePooling(strategy="mean", name=f"{self.name}_text_pooling")

        self.bbox_regressor = keras.layers.Dense(4, name=f"{self.name}_bbox_regressor")
        # DECISION plan-2026-10-14T062616-81938e81/D-003
        # Linear: 'confidence' carries region logits under its old name. Do
        # NOT put activation='sigmoid' back: the sigmoid belongs to the loss
        # (from_logits=True) or the caller, as for every class-like output of
        # layers/heads, and the argmax region pick is the same either way.
        # The bbox sigmoid in call() is a range map and stays. See
        # decisions.md D-003.
        self.confidence_scorer = keras.layers.Dense(
            1, name=f"{self.name}_confidence"
        )

    def build(self, input_shape: Union[Dict, Tuple, List]) -> None:
        """Explicitly build every sub-layer ``call()`` uses, in order.

        ``call()`` pools the text features to one query vector and TILES it
        across the vision regions. The fusion therefore sees two
        ``(B, N_regions, D)`` tensors. The text side keeps its own channel
        count, not a pooled-away one.

        The head has no post-fusion layers to build: its ``call()`` goes
        straight from ``fusion`` to ``confidence_scorer``.

        :param input_shape: Shape dict with ``vision_features`` (3-D, per-region)
            and ``text_features``.
        :type input_shape: Union[Dict, Tuple, List]
        """
        if self.built:
            return

        vision_shape = tuple(input_shape["vision_features"])
        text_shape = tuple(input_shape["text_features"])
        batch, num_regions = vision_shape[0], vision_shape[1]

        # Text is mean-pooled over its sequence axis then tiled to num_regions.
        text_tiled_shape = (batch, num_regions, text_shape[-1])

        fused_shape = self._build_fusion_stack([vision_shape, text_tiled_shape])
        self.text_pooling.build(text_shape)

        # confidence_scorer sees the per-region fused tensor; bbox_regressor sees
        # only the single top-scoring region, i.e. the same width without the
        # region axis.
        self.confidence_scorer.build(fused_shape)
        self.bbox_regressor.build((batch, fused_shape[-1]))

        super().build(input_shape)

    def call(
        self,
        inputs: Dict[str, keras.KerasTensor],
        training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        """
        Score every region, then box the top-scoring one.

        :param inputs: Dict with ``vision_features``, shaped
            ``(B, N_regions, D_vis)``, and ``text_features``, and optionally
            ``vision_mask`` and ``text_mask``.
        :type inputs: Dict[str, keras.KerasTensor]
        :param training: Keras training flag, forwarded to the fusion layer.
        :type training: Optional[bool]
        :return: Dict with ``bbox`` (sigmoid range map), ``confidence``
            (region logits) and ``grounded_features``.
        :rtype: Dict[str, keras.KerasTensor]
        :raises ValueError: If ``vision_features`` is not rank 3, or a
            ``text_mask`` is given for a rank-2 ``text_features``.
        """
        vision_features = inputs["vision_features"]
        vision_mask = inputs.get("vision_mask")

        if static_rank(vision_features, "vision_features", "VisualGroundingHead") != 3:
            raise ValueError(
                "VisualGroundingHead: vision_features must be rank 3, "
                "(batch, num_regions, width); the regions are scored one by one."
            )

        # Pool text features to a single query vector, over the real tokens.
        text_query = _pool_sequence(
            self.text_pooling, inputs, "text_features", "VisualGroundingHead")

        # Align text query with each visual region for fusion.
        num_regions = keras.ops.shape(vision_features)[1]
        text_expanded = keras.ops.expand_dims(text_query, axis=1)
        text_expanded = keras.ops.tile(text_expanded, [1, num_regions, 1])

        # Fuse each region with the text query. The output is [B, N_regions, D_fused].
        fused_per_region = self.fusion([vision_features, text_expanded], training=training)

        # Score each aligned region's features: one logit per region.
        region_scores = self.confidence_scorer(fused_per_region)
        # Drop the trailing size-1 axis, leaving [B, N_regions].
        region_scores = keras.ops.squeeze(region_scores, axis=-1)

        # Pick among the real regions only. The padded regions keep their
        # logits in 'confidence'; only the pick and what follows it ignore
        # them. Without a mask the scores are used as they are.
        pick_scores = region_scores
        if vision_mask is not None:
            pick_scores = fill_masked(region_scores, vision_mask)

        # Regress the bounding box from the top-scoring region's features.
        #
        # Gather one region per batch element with `keras.ops.take_along_axis`, not
        # with NumPy-style fancy indexing such as
        # `fused[batch_indices, top_indices]`. TF tensors reject that form
        # outright: "Only integers, slices (`:`), ellipsis, tf.newaxis and
        # scalar tf.int32/tf.int64 tensors are valid indices". It failed
        # eagerly, not only under tracing, which left this head dead on its
        # forward pass.
        top_indices = keras.ops.argmax(pick_scores, axis=1)
        gather_index = keras.ops.reshape(keras.ops.cast(top_indices, "int32"), (-1, 1, 1))
        gather_index = keras.ops.broadcast_to(
            gather_index, (keras.ops.shape(fused_per_region)[0], 1, keras.ops.shape(fused_per_region)[2])
        )
        top_features = keras.ops.squeeze(
            keras.ops.take_along_axis(fused_per_region, gather_index, axis=1), axis=1
        )
        bbox = self.bbox_regressor(top_features)

        return {
            "bbox": keras.ops.sigmoid(bbox),
            "confidence": region_scores,
            "grounded_features": top_features,
        }

    def compute_output_shape(self, input_shape: Dict) -> Dict[str, Tuple[Optional[int], ...]]:
        """Returns output-shape dict mirroring call() outputs.

        :param input_shape: Dict with ``vision_features`` ``(B, N_regions, D_vis)``.
        :return: Shapes for ``bbox``, ``confidence`` and ``grounded_features``.
        :rtype: Dict[str, Tuple[Optional[int], ...]]
        """
        vision_shape = input_shape["vision_features"]
        batch, num_regions = vision_shape[0], vision_shape[1]
        return {
            "bbox": (batch, 4),
            "confidence": (batch, num_regions),
            "grounded_features": (batch, self.hidden_dim),
        }
