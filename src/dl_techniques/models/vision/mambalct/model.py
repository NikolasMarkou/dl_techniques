"""MambaLCT single-object tracker with long-term context in a state space.

:class:`MambaLCT` takes a template and a clip of search frames and returns a
presence score and a box per frame, plus an updated context for the next
call. Instead of matching each frame to the template in isolation, it scans
the search tokens of the whole clip causally with a Mamba-style layer that
carries target-change cues forward through a few context tokens, then fuses
the template in with one cross-attention layer. Inputs must be square at the
configured ``template_size`` and ``search_size``. Boxes leave the model
sigmoid-normalized. ``pretrained=True`` raises ``NotImplementedError``; pass
a local checkpoint path instead.

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
from dl_techniques.utils.model_build import refuse_input_shape
from dl_techniques.utils.weight_transfer import load_weights_from_checkpoint
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
# Token width of the encoder output, SSM and fusion; not configurable.
EMBED_DIM = 512
CONTEXT_LEN = 1
#: Initializer of the fusion attention's three projections (``q``, ``kv``, ``proj``). It is the
#: layer's own ``kernel_initializer`` default, named here once so that each projection can be
#: given its own instance: the layer hands a single ``kernel_initializer`` instance to ``q`` and
#: ``kv``, and, without ``output_kernel_initializer``, to ``proj`` too, which then all start
#: from the same random stream. A test compares the class with the layer's default.
FUSION_ATTENTION_INITIALIZER = "glorot_uniform"

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.mambalct.model")
class MambaLCT(keras.Model):
    """Track one object through a search clip with SSM-carried context.

    The template and every search frame pass through one shared
    :class:`UcaEncoder`. :class:`ContextMambaLayer` scans the search tokens
    of all ``T`` frames with the context tokens and returns enhanced tokens
    plus an updated context. One cross-attention layer reads the template
    tokens into each enhanced frame. The detection head sees one mean-pooled
    token per frame. Boxes pass through a sigmoid in ``call``; scores come
    from the head's classification output with no activation added here.
    The token width is fixed at 512, so the last ``stage_dims`` entry must
    be 512.

    Search path:

    .. code-block:: text

        search [B, T, Hs, Ws, 3]
                 │
                 ▼
        reshape [B*T, Hs, Ws, 3]
                 │
                 ▼
        ┌─────────────────────┐
        │ uca_encoder         │  shared with template
        └─────────────────────┘
                 │  [B*T, N, D]
                 ▼
        reshape [B, T, N, D]
                 │
                 ▼
        ┌─────────────────────┐
        │ context_mamba       │◄── context [B, Nc, D], or zeros
        │                     │──► updated_context [B, Nc, D]
        └─────────────────────┘
                 │  [B, T, N, D]
                 ▼
        reshape [B*T, N, D]  (query)
                 │
                 ├─────────────────┐ residual
                 ▼                 │
        ┌─────────────────────┐    │
        │ fusion_attn         │◄───┼── key [B*T, Nt, D]
        └─────────────────────┘    │
                 │                 │
                 ▼                 │
                 + ◄───────────────┘
                 │
                 ▼
        ┌─────────────────────┐
        │ fusion_norm         │
        └─────────────────────┘
                 │  [B*T, N, D]
                 ▼
        mean over N, expand dims
                 │  [B*T, 1, 1, D]
                 ▼
        ┌─────────────────────┐
        │ detection_head      │
        └─────────────────────┘
                 │
           ┌─────┴──────────┐
           ▼                ▼
        classifications  regressions
        [B*T, 1, 1, 1]   [B*T, 1, 1, 4]
           │                │
           ▼                ▼
        reshape          reshape, sigmoid
           │                │
           ▼                ▼
        scores           boxes
        [B, T, 1]        [B, T, 4]

    D is embed_dim (512); Nc is context_len; N, Nt are token counts.

    Template key path:

    .. code-block:: text

        template [B, Ht, Wt, 3]
                 │
                 ▼
        ┌─────────────────────┐
        │ uca_encoder         │  shared with search
        └─────────────────────┘
                 │  [B, Nt, D]
                 ▼
        einsum with ones [B, T]
                 │  [B, T, Nt, D]
                 ▼
        reshape [B*T, Nt, D]  (key)

    Variants:

    .. code-block:: text

        variant       template_size  search_size
        mambalct-256  128            256
        mambalct-384  192            384

    :param template_size: Square template extent in pixels. Must be positive.
    :type template_size: int
    :param search_size: Square search extent in pixels. Must be positive.
    :type search_size: int
    :param context_len: Number of context tokens. Must be positive.
    :type context_len: int
    :param stage_dims: Encoder channel widths per stage. Defaults to
        ``[128, 256, 512]``. The last entry must be 512.
    :type stage_dims: List[int] or None
    :param stage_depths: Encoder blocks per stage. Defaults to ``[2, 2, 6]``.
    :type stage_depths: List[int] or None
    :param num_heads: Encoder heads per stage. Defaults to ``[4, 8, 8]``.
    :type num_heads: List[int] or None
    :param mlp_ratio: Encoder FFN expansion ratio.
    :type mlp_ratio: float
    :param d_state: SSM latent state size.
    :type d_state: int
    :param d_conv: SSM convolution kernel size.
    :type d_conv: int
    :param expand: SSM expansion factor.
    :type expand: int
    :param fusion_heads: Cross-attention fusion heads. Must divide 512.
    :type fusion_heads: int
    :param head_hidden_dim: Hidden width of the detection head. Must be
        positive.
    :type head_hidden_dim: int
    :param dropout_rate: Dropout rate for the encoder, fusion attention and
        detection head. ``context_mamba`` has none. Must be in ``[0, 1)``.
    :type dropout_rate: float
    :param **kwargs: Passed to ``keras.Model``.
    :ivar embed_dim: Token width, fixed at 512.
    :vartype embed_dim: int
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
        """Validate arguments and create the sub-layers.

        Arguments are documented on the class.

        :raises ValueError: If ``template_size``, ``search_size``,
            ``context_len`` or ``head_hidden_dim`` is not positive, the last
            ``stage_dims`` entry is not 512, 512 is not divisible by
            ``fusion_heads``, or ``dropout_rate`` is outside ``[0, 1)``, or if
            ``input_shape`` or ``input_dim`` is passed (the model is sized by
            ``template_size`` and ``search_size``).
        """
        refuse_input_shape(
            kwargs, type(self).__name__,
            "the model is sized by template_size and search_size")
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

        # Created first so its internal layers get deterministic names.
        # One class and one anchor give one score and one box per frame.
        self.detection_head = create_vision_head(
            VisionTaskType.DETECTION,
            num_classes=1,
            num_anchors=1,
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
        # DECISION plan-2026-10-26T125959-7af851e5/D-050: all three projections of the
        # fusion attention are given an initializer of their own. Left at the layer's
        # default, `q`, `kv` and `proj` hold ONE instance and start from the same random
        # stream (measured: 3 of 496 kernel pairs of the tiny model, r = 1.000000). Do not
        # drop `kv_kernel_initializer` or `output_kernel_initializer` to simplify the
        # call: nothing but the replay oracle sees the difference. See decisions.md.
        self.fusion_attn = create_attention_layer(
            "multi_head_cross",
            dim=self.embed_dim,
            num_heads=self.fusion_heads,
            dropout_rate=self.dropout_rate,
            kernel_initializer=FUSION_ATTENTION_INITIALIZER,
            kv_kernel_initializer=FUSION_ATTENTION_INITIALIZER,
            output_kernel_initializer=FUSION_ATTENTION_INITIALIZER,
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
        """Build the encoder, SSM, fusion and head sub-layers.

        The shared encoder is built once, on the template shape.
        ``context_mamba`` is built with a ``(None, context_len, D)`` context
        whatever the third input shape says, and ``detection_head`` on
        ``(None, 1, 1, D)``. Returns at once if the model is already built.

        :param input_shape: Sequence of ``(template, search[, context])``
            shapes. Any other value falls back to ``template_size`` and
            ``search_size`` with ``None`` batch and frame axes.
        """
        if self.built:
            return

        if isinstance(input_shape, (tuple, list)) and len(input_shape) >= 2:
            template_shape = input_shape[0]
            search_shape = input_shape[1]
        else:
            template_shape = (None, self.template_size, self.template_size, 3)
            search_shape = (None, None, self.search_size, self.search_size, 3)

        if not self.encoder.built:
            self.encoder.build(template_shape)

        flat_search_shape = (None, search_shape[2], search_shape[3], 3)
        # No-op: the shared encoder is already built on the template shape.
        if not self.encoder.built:
            self.encoder.build(flat_search_shape)

        search_tokens_shape = self.encoder.compute_output_shape(flat_search_shape)
        tokens_per_frame = search_tokens_shape[1]
        search_flow_shape = (None, search_shape[1], tokens_per_frame, self.embed_dim)

        if not self.context_mamba.built:
            context_shape = (None, self.context_len, self.embed_dim)
            self.context_mamba.build([search_flow_shape, context_shape])

        if not self.fusion_attn.built:
            self.fusion_attn.build(search_tokens_shape)
        if not self.fusion_norm.built:
            self.fusion_norm.build(search_tokens_shape)

        # call() feeds the head pooled tokens reshaped to (B*T, 1, 1, D).
        pooled_shape = (None, 1, 1, self.embed_dim)
        if not self.detection_head.built:
            self.detection_head.build(pooled_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: Union[Tuple[Any, Any, Any], List[Any]],
        training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        """Score and box every search frame against the template.

        :param inputs: ``(template, search)`` or ``(template, search,
            context)``. ``template`` is ``(B, Ht, Wt, 3)`` with
            ``Ht == Wt == template_size``; ``search`` is
            ``(B, T, Hs, Ws, 3)`` with ``Hs == Ws == search_size``;
            ``context`` is ``(B, Nc, D)`` and defaults to zeros.
        :param training: Forwarded to every sub-layer.
        :return: Dict with ``scores`` ``(B, T, 1)`` taken from the head's
            classification output with no activation added here, ``boxes``
            ``(B, T, 4)`` as sigmoid-normalized ``(cx, cy, w, h)`` in
            search-crop coordinates, and ``updated_context`` ``(B, Nc, D)``.
        :raises ValueError: If ``inputs`` is not a 2- or 3-element tuple or
            list, or a static template or search extent differs from the
            configured size.
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
        # -1 keeps the reshape traceable with symbolic batch and frame axes.
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
        # Tile over T with ones taken from search, so a symbolic T still traces;
        # ops.tile needs static repeats and fails the build_from_config trace.
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
        # The head takes spatial (B, H, W, C) input.
        pooled_spatial = ops.expand_dims(ops.expand_dims(pooled, axis=1), axis=1)
        det_outputs = self.detection_head(pooled_spatial, training=training)
        scores = ops.reshape(det_outputs['classifications'], (batch, frames, 1))
        boxes = ops.reshape(det_outputs['regressions'], (batch, frames, 4))
        boxes = ops.sigmoid(boxes)
        return {
            "scores": scores,
            "boxes": boxes,
            "updated_context": updated_context,
        }

    def _check_spatial_contract(
        self, template: keras.KerasTensor, search: keras.KerasTensor
    ) -> None:
        """Raise when a static input extent differs from the configured size.

        Extents that are ``None`` at trace time are skipped.

        :param template: Template input ``(B, Ht, Wt, 3)``.
        :param search: Search input ``(B, T, Hs, Ws, 3)``.
        :raises ValueError: If a static ``Hs`` or ``Ws`` differs from
            ``search_size``, or a static ``Ht`` or ``Wt`` differs from
            ``template_size``.
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
        """Return the constructor arguments for serialization.

        ``embed_dim`` is not included; it is fixed at 512.

        :return: The base ``keras.Model`` config updated with every
            constructor argument.
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
        # __init__ rebuilds every sub-layer, detection_head included, from this.
        return config

    @classmethod
    def from_variant(
        cls,
        variant: str,
        pretrained: Union[bool, str] = False,
        **kwargs: Any,
    ) -> "MambaLCT":
        """Create a model from a named variant, optionally loading weights.

        Keyword overrides take precedence over the variant's values. A string
        ``pretrained`` builds the model on ``(template, search, context)``
        shapes with ``None`` batch and frame axes, then loads the checkpoint
        with ``strict=True``. Errors from ``load_weights_from_checkpoint``
        propagate.

        :param variant: ``"mambalct-256"`` or ``"mambalct-384"``.
        :param pretrained: Path to a local checkpoint, or ``False`` to skip
            loading. ``True`` raises.
        :param **kwargs: Constructor arguments that override the variant's
            values.
        :return: A MambaLCT instance.
        :raises ValueError: If the variant is unknown or the constructor
            rejects the merged arguments.
        :raises NotImplementedError: If ``pretrained`` is ``True``.
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
    """Create a :class:`MambaLCT` through :meth:`MambaLCT.from_variant`.

    :param variant: ``"mambalct-256"`` or ``"mambalct-384"``.
    :param pretrained: Path to a local checkpoint, or ``False``. ``True``
        raises.
    :param **kwargs: Constructor arguments forwarded to :class:`MambaLCT`.
    :return: A MambaLCT instance.
    :raises ValueError: If the variant is unknown or an argument is invalid.
    :raises NotImplementedError: If ``pretrained`` is ``True``.
    """
    return MambaLCT.from_variant(variant, pretrained=pretrained, **kwargs)

# ---------------------------------------------------------------------