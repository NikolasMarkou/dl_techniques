"""MambaLCT visual tracker: long-term context via a selective state space.

Tracking is reframed as appearance-plus-change modeling: a shared
hierarchical encoder (:class:`UcaEncoder`) maps the template and each
search frame of a clip into one token space, a unidirectional
:class:`ContextMambaLayer` scans the search flow and rolls target-change
cues from the first frame to the current one into the context tokens
(``Y_i^T = C H_i^T`` updating ``c_p``), a cross-attention fuses template
information into the enhanced search tokens, and a per-frame global head
emits a presence score plus a box in normalized search-crop coordinates.

This package ships the ``mambalct-256`` / ``mambalct-384`` variants, which
share the backbone configuration and differ in template/search resolution
(paper Tab. 1: both HiViT-B, 72M). The v1 head predicts one box per frame
from mean-pooled fused tokens; a dense OSTrack-style map head is deferred
and recorded in the README. ``pretrained=True`` raises
``NotImplementedError``; pass a local ``.keras`` path instead.

References:
    - Li et al., 2024. MambaLCT: Boosting Tracking via Long-term Context
      State Space Model. (https://arxiv.org/abs/2412.13615)
    - Ye et al., 2022. Joint Feature Learning and Relation Modeling for
      Tracking: A One-Stream Framework (OSTrack). (https://arxiv.org/abs/2203.11922)
    - Gu and Dao, 2023. Mamba: Linear-Time Sequence Modeling with Selective
      State Spaces. (https://arxiv.org/abs/2312.00752)
"""

import keras
from keras import ops
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.weight_transfer import load_weights_from_checkpoint
from dl_techniques.utils.model_build import materialize_sublayers
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.layers.attention.factory import create_attention_layer
from dl_techniques.layers.norms.factory import create_normalization_layer
from dl_techniques.layers.ssm.context_mamba import ContextMambaLayer
from dl_techniques.layers.heads.vision import create_vision_head, VisionTaskType
from .uca_encoder import UcaEncoder

# ---------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------

TEMPLATE_SIZE_256 = 128
SEARCH_SIZE_256 = 256
TEMPLATE_SIZE_384 = 192
SEARCH_SIZE_384 = 384
EMBED_DIM = 512
CONTEXT_LEN = 1

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.mambalct.model")
class MambaLCT(keras.Model):
    """Single-object tracker with SSM-carried long-term context.

    One shared :class:`UcaEncoder` embeds the template and every search
    frame; :class:`ContextMambaLayer` scans the search flow with the
    context tokens prepended (so the causal scan carries their cues
    forward into every frame) and one learnable bridge token appended,
    whose output becomes the history-aggregated context update;
    cross-attention reads the template into each enhanced frame; a global
    head predicts per-frame scores and boxes.

    :param template_size: Square template extent in pixels. Must be positive.
    :type template_size: int
    :param search_size: Square search extent in pixels. Must be positive.
    :type search_size: int
    :param context_len: Number of context tokens. Must be positive.
    :type context_len: int
    :param stage_dims: Encoder channel widths per stage.
    :type stage_dims: List[int] or None
    :param stage_depths: Encoder blocks per stage.
    :type stage_depths: List[int] or None
    :param num_heads: Encoder heads per stage.
    :type num_heads: List[int] or None
    :param mlp_ratio: Encoder FFN expansion ratio.
    :type mlp_ratio: float
    :param d_state: SSM latent state size.
    :type d_state: int
    :param d_conv: SSM convolution kernel size.
    :type d_conv: int
    :param expand: SSM expansion factor.
    :type expand: int
    :param fusion_heads: Cross-attention fusion heads. Must divide embed dim.
    :type fusion_heads: int
    :param head_hidden_dim: Hidden width of the score/box MLPs.
    :type head_hidden_dim: int
    :param dropout_rate: Dropout rate.
    :type dropout_rate: float
    :param kwargs: Passthrough to ``keras.Model``.
    """

    MODEL_VARIANTS = {
        "mambalct-256": {
            "template_size": TEMPLATE_SIZE_256,
            "search_size": SEARCH_SIZE_256,
            "description": "MambaLCT-256: 256px search, 128px template",
        },
        "mambalct-384": {
            "template_size": TEMPLATE_SIZE_384,
            "search_size": SEARCH_SIZE_384,
            "description": "MambaLCT-384: 384px search, 192px template",
        },
    }

    def __init__(
        self,
        template_size: int = TEMPLATE_SIZE_256,
        search_size: int = SEARCH_SIZE_256,
        context_len: int = CONTEXT_LEN,
        stage_dims: Optional[List[int]] = None,
        stage_depths: Optional[List[int]] = None,
        num_heads: Optional[List[int]] = None,
        mlp_ratio: float = 4.0,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        fusion_heads: int = 8,
        head_hidden_dim: int = 256,
        dropout_rate: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if template_size <= 0:
            raise ValueError(f"template_size must be positive, got {template_size}")
        if search_size <= 0:
            raise ValueError(f"search_size must be positive, got {search_size}")
        if context_len <= 0:
            raise ValueError(f"context_len must be positive, got {context_len}")
        dims = [128, 256, EMBED_DIM] if stage_dims is None else list(stage_dims)
        if dims[-1] != EMBED_DIM:
            raise ValueError(
                f"final stage_dim must equal {EMBED_DIM} to match the "
                f"context-token width, got {dims[-1]}"
            )
        if EMBED_DIM % fusion_heads != 0:
            raise ValueError(
                f"embed dim ({EMBED_DIM}) must be divisible by fusion_heads "
                f"({fusion_heads})"
            )
        if head_hidden_dim <= 0:
            raise ValueError(
                f"head_hidden_dim must be positive, got {head_hidden_dim}"
            )
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(
                f"dropout_rate must be in [0, 1), got {dropout_rate}"
            )

        self.template_size = template_size
        self.search_size = search_size
        self.context_len = context_len
        self.stage_dims = dims
        self.stage_depths = (
            [2, 2, 6] if stage_depths is None else list(stage_depths)
        )
        self.num_heads = [4, 8, 8] if num_heads is None else list(num_heads)
        self.mlp_ratio = mlp_ratio
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.fusion_heads = fusion_heads
        self.head_hidden_dim = head_hidden_dim
        self.dropout_rate = dropout_rate
        self.embed_dim = EMBED_DIM

        # Detection head using factory (replaces custom score/box MLPs)
        # Created FIRST to ensure its internal layers get deterministic names
        self.detection_head = create_vision_head(
            VisionTaskType.DETECTION,
            num_classes=1,  # binary: object / no object
            num_anchors=1,  # one prediction per frame
            bbox_dims=4,
            input_format="spatial",
            hidden_dim=self.head_hidden_dim,
            use_attention=False,
            use_ffn=False,
            dropout_rate=self.dropout_rate,
            name="detection_head"
        )

        self.encoder = UcaEncoder(
            stage_dims=self.stage_dims,
            stage_depths=self.stage_depths,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            dropout_rate=self.dropout_rate,
            name="uca_encoder",
        )
        self.context_mamba = ContextMambaLayer(
            d_model=self.embed_dim,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
            name="context_mamba",
        )
        # Routed through the attention factory (strict kwargs), not hand-rolled.
        self.fusion_attn = create_attention_layer(
            "multi_head_cross",
            dim=self.embed_dim,
            num_heads=self.fusion_heads,
            dropout_rate=self.dropout_rate,
            name="fusion_attn",
        )
        self.fusion_norm = create_normalization_layer(
            "layer_norm", epsilon=1e-5, name="fusion_norm"
        )

        logger.info(
            f"Created MambaLCT (template={template_size}, "
            f"search={search_size}, context_len={context_len})"
        )

    def build(self, input_shape: Any) -> None:
        """Materialize the encoder, SSM, fusion, and head sub-layers.

        :param input_shape: Nest of ``(template, search, context)`` shapes.
        """
        if self.built:
            return
        
        # Parse input shapes for proper sub-layer building
        if isinstance(input_shape, (tuple, list)) and len(input_shape) >= 2:
            template_shape = input_shape[0]
            search_shape = input_shape[1]
        else:
            template_shape = (None, self.template_size, self.template_size, 3)
            search_shape = (None, None, self.search_size, self.search_size, 3)
        
        # Build encoder first
        if not self.encoder.built:
            self.encoder.build(template_shape)
        
        # Build search encoder
        encoder_out_shape = self.encoder.compute_output_shape(template_shape)
        flat_search_shape = (None, search_shape[2], search_shape[3], 3)
        if not self.encoder.built:
            self.encoder.build(flat_search_shape)
        
        search_tokens_shape = self.encoder.compute_output_shape(flat_search_shape)
        tokens_per_frame = search_tokens_shape[1]
        search_flow_shape = (None, search_shape[1], tokens_per_frame, self.embed_dim)
        
        # Build context_mamba
        if not self.context_mamba.built:
            context_shape = (None, self.context_len, self.embed_dim)
            self.context_mamba.build([search_flow_shape, context_shape])
        
        # Build fusion
        if not self.fusion_attn.built:
            self.fusion_attn.build(search_tokens_shape)
        if not self.fusion_norm.built:
            self.fusion_norm.build(search_tokens_shape)
        
        # Build detection head on the pooled feature shape (B, 1, 1, embed_dim)
        # The call() reshapes pooled to (B, 1, 1, D) before passing to detection_head
        pooled_shape = (None, 1, 1, self.embed_dim)
        if not self.detection_head.built:
            self.detection_head.build(pooled_shape)
        
        super().build(input_shape)

    def call(
        self,
        inputs: Union[Tuple[Any, Any, Any], List[Any]],
        training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        """Track a template against a search clip with carried context.

        :param inputs: Pair ``(template, search)`` or triple
            ``(template, search, context)`` with template
            ``(B, Ht, Wt, 3)``, search ``(B, T, Hs, Ws, 3)`` with
            ``Hs == Ws == search_size``, context ``(B, Nc, D)``.
            A missing context becomes zeros.
        :param training: Training flag forwarded to sub-layers.
        :return: Dict with ``scores`` ``(B, T, 1)``, ``boxes`` ``(B, T, 4)``
            in normalized search-crop ``(cx, cy, w, h)``, and
            ``updated_context`` ``(B, Nc, D)``.
        :raises ValueError: If the inputs are not a pair/triple, or a
            statically-known search/template extent disagrees with the
            configured sizes.
        """
        if isinstance(inputs, (tuple, list)) and len(inputs) == 2:
            template, search, context = inputs[0], inputs[1], None
        elif isinstance(inputs, (tuple, list)) and len(inputs) == 3:
            template, search, context = inputs[0], inputs[1], inputs[2]
        else:
            raise ValueError(
                "MambaLCT expects a (template, search) pair or a "
                "(template, search, context) triple"
            )
        self._check_spatial_contract(template, search)
        batch = ops.shape(template)[0]
        frames = ops.shape(search)[1]
        search_h = ops.shape(search)[2]
        search_w = ops.shape(search)[3]

        template_tokens = self.encoder(template, training=training)
        # Dynamic reshape: no Python arithmetic on shape tensors, so this
        # traces with symbolic (None) batch/frames axes (build_from_config,
        # inference with carried context).
        flat_search = ops.reshape(search, (-1, search_h, search_w, 3))
        search_tokens = self.encoder(flat_search, training=training)
        tokens_per_frame = ops.shape(search_tokens)[1]
        search_flow = ops.reshape(
            search_tokens, (batch, frames, tokens_per_frame, self.embed_dim)
        )

        if context is None:
            context = ops.zeros(
                (batch, self.context_len, self.embed_dim),
                dtype=search_flow.dtype,
            )

        enhanced, updated_context = self.context_mamba(
            [search_flow, context], training=training
        )
        flat_frames = ops.shape(search_tokens)[0]
        query = ops.reshape(
            enhanced, (flat_frames, tokens_per_frame, self.embed_dim)
        )
        # Broadcast tiling without materializing a shaped canvas: ones
        # derived from the search tensor carry the dynamic (B, T) shape, so
        # this traces with symbolic frames (ops.tile/zeros need static
        # repeats/extents and fail the build_from_config trace instead).
        template_tokens_len = ops.shape(template_tokens)[1]
        frame_ones = ops.ones_like(search[:, :, 0, 0, 0])
        template_tiled = ops.einsum(
            "bnd,bt->btnd", template_tokens, frame_ones
        )
        key = ops.reshape(
            template_tiled,
            (flat_frames, template_tokens_len, self.embed_dim),
        )
        fused = self.fusion_attn(query, key, training=training)
        fused = self.fusion_norm(
            query + fused, training=training
        )
        pooled = ops.mean(fused, axis=1)
        # DetectionHead expects spatial input (B, H, W, C) but we have (B, D)
        # Reshape to (B, 1, 1, D) for spatial format
        pooled_spatial = ops.expand_dims(ops.expand_dims(pooled, axis=1), axis=1)
        det_outputs = self.detection_head(pooled_spatial, training=training)
        # det_outputs: {'classifications': (B, 1, 1, 1), 'regressions': (B, 1, 1, 4)}
        scores = ops.reshape(det_outputs['classifications'], (batch, frames, 1))
        boxes = ops.reshape(det_outputs['regressions'], (batch, frames, 4))
        # Apply sigmoid to boxes as the original did
        boxes = ops.sigmoid(boxes)
        return {
            "scores": scores,
            "boxes": boxes,
            "updated_context": updated_context,
        }

    def _check_spatial_contract(
        self, template: keras.KerasTensor, search: keras.KerasTensor
    ) -> None:
        """Fail loudly when a statically-known extent breaks the geometry.

        The encoder downsamples by a fixed stride and the head reads
        ``search_size``-shaped frames; a mismatched static extent would
        otherwise die in an opaque reshape. Dynamic (``None``) extents skip
        the check — they cannot be judged at trace time.

        :param template: Template input ``(B, Ht, Wt, 3)``.
        :param search: Search input ``(B, T, Hs, Ws, 3)``.
        :raises ValueError: On a known-bad static extent.
        """
        static_search = (search.shape[2], search.shape[3])
        for extent, name in zip(static_search, ("Hs", "Ws")):
            if extent is not None and extent != self.search_size:
                raise ValueError(
                    f"search {name} ({extent}) must equal search_size "
                    f"({self.search_size})"
                )
        static_template = (template.shape[1], template.shape[2])
        for extent, name in zip(static_template, ("Ht", "Wt")):
            if extent is not None and extent != self.template_size:
                raise ValueError(
                    f"template {name} ({extent}) must equal template_size "
                    f"({self.template_size})"
                )

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration for serialization.

        :return: Dictionary containing all constructor arguments.
        """
        config = super().get_config()
        config.update({
            "template_size": self.template_size,
            "search_size": self.search_size,
            "context_len": self.context_len,
            "stage_dims": list(self.stage_dims),
            "stage_depths": list(self.stage_depths),
            "num_heads": list(self.num_heads),
            "mlp_ratio": self.mlp_ratio,
            "d_state": self.d_state,
            "d_conv": self.d_conv,
            "expand": self.expand,
            "fusion_heads": self.fusion_heads,
            "head_hidden_dim": self.head_hidden_dim,
            "dropout_rate": self.dropout_rate,
        })
        # Note: detection_head is serialized as a sub-layer via Keras standard mechanism
        return config

    @classmethod
    def from_variant(
        cls,
        variant: str,
        pretrained: Union[bool, str] = False,
        **kwargs: Any,
    ) -> "MambaLCT":
        """Create a MambaLCT model from a predefined variant.

        :param variant: One of ``"mambalct-256"`` / ``"mambalct-384"``.
        :param pretrained: Local checkpoint path to load, or True to raise
            since no weights ship here.
        :param kwargs: Overrides for the variant defaults.
        :return: A MambaLCT instance.
        :raises ValueError: If the variant is unknown.
        :raises NotImplementedError: If ``pretrained`` is True.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )
        config = cls.MODEL_VARIANTS[variant].copy()
        config.pop("description", "")
        config.update(kwargs)
        logger.info(f"Creating MambaLCT variant '{variant}'")
        model = cls(**config)
        if pretrained:
            if isinstance(pretrained, str):
                if not model.built:
                    model.build([
                        (None, model.template_size, model.template_size, 3),
                        (None, None, model.search_size, model.search_size, 3),
                        (None, model.context_len, model.embed_dim),
                    ])
                report = load_weights_from_checkpoint(
                    target=model, ckpt_path=pretrained, strict=True
                )
                logger.info(report.summary_string())
            else:
                raise NotImplementedError(
                    "No pretrained MambaLCT weights are distributed with "
                    f"dl_techniques (variant '{variant}'). Pass a local "
                    "checkpoint instead: from_variant(..., pretrained='/path/"
                    "to/weights.keras')."
                )
        return model


# ---------------------------------------------------------------------


def create_mambalct(
    variant: str = "mambalct-256",
    pretrained: Union[bool, str] = False,
    **kwargs: Any,
) -> MambaLCT:
    """Create a :class:`MambaLCT` tracker with no logic beyond construction.

    :param variant: Variant name, ``"mambalct-256"`` or ``"mambalct-384"``.
    :param pretrained: Local checkpoint path, or True to raise.
    :param kwargs: Overrides forwarded to :class:`MambaLCT`.
    :return: A MambaLCT instance.
    """
    return MambaLCT.from_variant(variant, pretrained=pretrained, **kwargs)

# ---------------------------------------------------------------------
