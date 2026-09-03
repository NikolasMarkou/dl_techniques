"""A configurable Vision Transformer encoder, built by :class:`VisionEncoder`.

An input image is split into a grid of non-overlapping patches, each patch
linearly projected into a `D`-dimensional embedding, and a learnable CLS
token optionally prepended: `z0 = [x_class; E*p1; ...; E*p_N] + E_pos`. Patch
embedding, attention type, normalization type and position, and FFN type are
all constructor arguments routed through factory components, so one class
covers architectures from the standard ViT to SigLIP's two-stage patch
embedder or a RMSNorm + SwiGLU variant.

References:
    - Dosovitskiy et al., 2020. An Image is Worth 16x16 Words: Transformers
      for Image Recognition at Scale.
    - Vaswani et al., 2017. Attention Is All You Need.
    - Zhai et al., 2023. Sigmoid Loss for Language Image Pre-Training.
"""

import keras
from keras import ops, layers, initializers, regularizers
from typing import Optional, Union, Tuple, Dict, Any, Literal, Callable, get_args

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ...initializers import clone_initializer
from ..embedding import create_embedding_layer
from ..norms import create_normalization_layer
from .transformer import (
    TransformerLayer,
    NormalizationType,
    NormalizationPositionType,
    AttentionType,
    FFNType
)
from ..sequence_pooling import SequencePooling, PoolingStrategy
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------
# Type definitions for enhanced type safety
# ---------------------------------------------------------------------

PatchEmbedType = Literal['linear', 'siglip', 'conv', 'hybrid']

# DECISION plan-2026-07-31T132403-b3f540cb/D-003: no mask-incompatible-mode
# allowlist here (supersedes plan-2026-07-31T042809-ddc92265/D-013) -- masked-patch isolation for weighted/top_k pooling is now fixed inside layers/sequence_pooling/, not guarded per-caller. See decisions.md.

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.transformers.vision_encoder")
class VisionEncoder(keras.layers.Layer):
    """
    General-purpose configurable vision encoder using factory-based components.

    Converts images into patch sequences, adds optional CLS token and
    positional embeddings, processes through a configurable TransformerLayer
    stack, and pools output features. Factory patterns allow replicating
    architectures from standard ViT to SigLIP, DeiT, and modern variants
    with RMSNorm + SwiGLU.

    Architecture:

    .. code-block:: text

        ┌──────────────────────────────────────────┐
        │  Input Image (B, H, W, C)                │
        └──────────────────┬───────────────────────┘
                           ▼
        ┌──────────────────────────────────────────┐
        │  Patch Embedding (linear/siglip/conv)    │
        │  ─► Reshape (B, num_patches, embed_dim)  │
        └──────────────────┬───────────────────────┘
                           ▼
        ┌──────────────────────────────────────────┐
        │  [CLS Token] + Positional Embedding      │
        └──────────────────┬───────────────────────┘
                           ▼
        ┌──────────────────────────────────────────┐
        │  TransformerLayer x depth                │
        └──────────────────┬───────────────────────┘
                           ▼
        ┌──────────────────────────────────────────┐
        │  [Final Normalization] (pre-norm only)   │
        └──────────────────┬───────────────────────┘
                           ▼
        ┌──────────────────────────────────────────┐
        │  SequencePooling (cls/mean/max/none)     │
        └──────────────────┬───────────────────────┘
                           ▼
        ┌──────────────────────────────────────────┐
        │  Output Features                         │
        └──────────────────────────────────────────┘

    :param img_size: Input image spatial size. Default: 224.
    :type img_size: int
    :param patch_size: Patch side length. Default: 16.
    :type patch_size: int
    :param embed_dim: Embedding dimension. Default: 768.
    :type embed_dim: int
    :param depth: Number of transformer layers. Default: 12.
    :type depth: int
    :param num_heads: Number of attention heads. Default: 12.
    :type num_heads: int
    :param mlp_ratio: MLP expansion ratio. Default: 4.0.
    :type mlp_ratio: float
    :param patch_embed_type: Patch embedding strategy. Default: ``'linear'``.
    :type patch_embed_type: PatchEmbedType
    :param attention_type: Attention mechanism. Default: ``'multi_head'``.
    :type attention_type: AttentionType
    :param normalization_type: Normalization type. Default: ``'layer_norm'``.
    :type normalization_type: NormalizationType
    :param normalization_position: ``'pre'`` or ``'post'``. Default: ``'post'``.
    :type normalization_position: NormalizationPositionType
    :param ffn_type: FFN architecture. Default: ``'mlp'``.
    :type ffn_type: FFNType
    :param use_cls_token: Prepend a CLS token. Default: True.
    :type use_cls_token: bool
    :param output_mode: Pooling strategy. Default: ``'cls'``.
    :type output_mode: PoolingStrategy
    :param dropout_rate: General dropout. Default: 0.0.
    :type dropout_rate: float
    :param attention_dropout_rate: Attention dropout. Default: 0.0.
    :type attention_dropout_rate: float
    :param pos_dropout_rate: Positional embedding dropout. Default: 0.0.
    :type pos_dropout_rate: float
    :param stochastic_depth_rate: Drop-path rate. Default: 0.0.
    :type stochastic_depth_rate: float
    :param activation: FFN activation. Default: ``'gelu'``.
    :type activation: Union[str, Callable]
    :param use_bias: Whether layers use bias. Default: True.
    :type use_bias: bool
    :param kernel_initializer: Kernel weight initializer.
    :type kernel_initializer: Union[str, initializers.Initializer]
    :param bias_initializer: Bias weight initializer.
    :type bias_initializer: Union[str, initializers.Initializer]
    :param kernel_regularizer: Kernel weight regularizer.
    :type kernel_regularizer: Optional[regularizers.Regularizer]
    :param bias_regularizer: Bias weight regularizer.
    :type bias_regularizer: Optional[regularizers.Regularizer]
    :param attention_args: Custom attention layer arguments.
    :type attention_args: Optional[Dict[str, Any]]
    :param norm_args: Custom normalization layer arguments.
    :type norm_args: Optional[Dict[str, Any]]
    :param ffn_args: Custom FFN layer arguments.
    :type ffn_args: Optional[Dict[str, Any]]
    :param patch_embed_args: Custom patch embedding arguments.
    :type patch_embed_args: Optional[Dict[str, Any]]
    :param kwargs: Additional keyword arguments for the base Layer.
    :type kwargs: Any

    :raises ValueError: If dimension parameters are invalid.
    """

    def __init__(
            self,
            img_size: int = 224,
            patch_size: int = 16,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            patch_embed_type: PatchEmbedType = 'linear',
            attention_type: AttentionType = 'multi_head',
            normalization_type: NormalizationType = 'layer_norm',
            normalization_position: NormalizationPositionType = 'post',
            ffn_type: FFNType = 'mlp',
            use_cls_token: bool = True,
            output_mode: PoolingStrategy = 'cls',
            dropout_rate: float = 0.0,
            attention_dropout_rate: float = 0.0,
            pos_dropout_rate: float = 0.0,
            stochastic_depth_rate: float = 0.0,
            activation: Union[str, Callable] = 'gelu',
            use_bias: bool = True,
            kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            bias_regularizer: Optional[regularizers.Regularizer] = None,
            attention_args: Optional[Dict[str, Any]] = None,
            norm_args: Optional[Dict[str, Any]] = None,
            ffn_args: Optional[Dict[str, Any]] = None,
            patch_embed_args: Optional[Dict[str, Any]] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if img_size <= 0:
            raise ValueError(f"img_size must be positive, got {img_size}")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if img_size % patch_size != 0:
            raise ValueError(
                f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"
            )
        # Multi-stage stems reach patch_size as a product of strides, and
        # integer division silently rounds -- without this check the encoder builds, then dies in an opaque reshape at the first forward.
        _stem_stride_divisor = {'siglip': 2, 'conv': 4}.get(patch_embed_type)
        if _stem_stride_divisor and patch_size % _stem_stride_divisor != 0:
            raise ValueError(
                f"patch_embed_type='{patch_embed_type}' needs a patch_size "
                f"divisible by {_stem_stride_divisor}, got {patch_size}: its "
                f"multi-stage stem's total stride would be "
                f"{_stem_stride_divisor * (patch_size // _stem_stride_divisor)} "
                f"instead of {patch_size}."
            )
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        if mlp_ratio <= 0.0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout must be between 0 and 1, got {dropout_rate}")
        if not (0.0 <= attention_dropout_rate <= 1.0):
            raise ValueError(f"attention_dropout must be between 0 and 1, got {attention_dropout_rate}")
        if not (0.0 <= pos_dropout_rate <= 1.0):
            raise ValueError(f"pos_dropout must be between 0 and 1, got {pos_dropout_rate}")
        # H-03: output_mode forwards to SequencePooling(strategy=); without this
        # check a typo failed later, inside the pooling layer. Derive the legal set from PoolingStrategy rather than restating it here.
        _legal_output_modes = get_args(PoolingStrategy)
        if output_mode not in _legal_output_modes:
            raise ValueError(
                f"output_mode must be one of {list(_legal_output_modes)}, "
                f"got {output_mode!r}"
            )
        if not use_cls_token and output_mode == 'cls':
            raise ValueError("output_mode='cls' requires use_cls_token=True")

        # Store configuration parameters for serialization
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.patch_embed_type = patch_embed_type
        self.attention_type = attention_type
        self.normalization_type = normalization_type
        self.normalization_position = normalization_position
        self.ffn_type = ffn_type
        self.use_cls_token = use_cls_token
        self.output_mode = output_mode
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.pos_dropout_rate = pos_dropout_rate
        self.stochastic_depth_rate = stochastic_depth_rate
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.attention_args = attention_args or {}
        self.norm_args = norm_args or {}
        self.ffn_args = ffn_args or {}
        self.patch_embed_args = patch_embed_args or {}

        # Computed properties
        self.num_patches = (img_size // patch_size) ** 2
        self.seq_len = self.num_patches + (1 if use_cls_token else 0)
        self.intermediate_size = int(embed_dim * mlp_ratio)

        # Create sub-layers in __init__ (modern Keras 3 pattern)

        # Create patch embedding using factory pattern
        self.patch_embed = self._create_patch_embedding()

        # Create positional embedding using factory
        self.pos_embed = create_embedding_layer(
            'positional_learned',
            max_seq_len=self.seq_len,
            dim=self.embed_dim,
            dropout_rate=self.pos_dropout_rate,
            name="pos_embed"
        )

        # Create transformer layers using factory components
        self.transformer_layers = []
        for i in range(self.depth):
            # Calculate stochastic depth rate (linearly increasing)
            layer_drop_rate = self.stochastic_depth_rate * i / max(1, self.depth - 1)

            layer = TransformerLayer(
                hidden_size=self.embed_dim,
                num_heads=self.num_heads,
                intermediate_size=self.intermediate_size,
                attention_type=self.attention_type,
                attention_args=self.attention_args,
                normalization_type=self.normalization_type,
                normalization_position=self.normalization_position,
                attention_norm_args=self.norm_args,
                ffn_norm_args=self.norm_args,
                ffn_type=self.ffn_type,
                ffn_args=self.ffn_args,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                use_stochastic_depth=self.stochastic_depth_rate > 0.0,
                stochastic_depth_rate=layer_drop_rate,
                activation=self.activation,
                use_bias=self.use_bias,
                # DECISION plan-2026-08-23T091307-9a110062/D-560: every block gets
                # its own clone_initializer copy, never a shared instance -- a shared one replays the identical draw at every same-shape kernel. See decisions.md.
                kernel_initializer=clone_initializer(self.kernel_initializer),
                bias_initializer=clone_initializer(self.bias_initializer),
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name=f"transformer_layer_{i}"
            )
            self.transformer_layers.append(layer)

        # Create final normalization layer (only for pre-norm)
        self.norm = None
        if self.normalization_position == 'pre':
            self.norm = create_normalization_layer(
                self.normalization_type,
                name="final_norm",
                **self.norm_args
            )

        # Create pooling layer using SequencePooling
        # For mean and max pooling with CLS token, we exclude position 0
        exclude_positions = [0] if (use_cls_token and output_mode in ['mean', 'max']) else []

        self.pooling_layer = SequencePooling(
            strategy=output_mode,
            exclude_positions=exclude_positions,
            name='output_pooling'
        )

        # Create CLS token weight if needed (shape is independent of input)
        self.cls_token = None
        if self.use_cls_token:
            self.cls_token = self.add_weight(
                name="cls_token",
                shape=(1, 1, self.embed_dim),
                initializer="zeros",
                trainable=True,
            )

    def _create_patch_embedding(self) -> keras.layers.Layer:
        """Create patch embedding layer based on the specified type.

        :return: Patch embedding layer.
        :rtype: keras.layers.Layer
        """
        base_args = {
            # D-560: a clone per patch-embed stack -- 'hybrid'/'overlapping'
            # build several convs from this one dict.
            'kernel_initializer': clone_initializer(self.kernel_initializer),
            'bias_initializer': clone_initializer(self.bias_initializer),
            'kernel_regularizer': self.kernel_regularizer,
            'bias_regularizer': self.bias_regularizer,
            'use_bias': self.use_bias
        }
        base_args.update(self.patch_embed_args)

        if self.patch_embed_type == 'linear':
            # Standard ViT-style linear patch embedding
            return layers.Conv2D(
                filters=self.embed_dim,
                kernel_size=self.patch_size,
                strides=self.patch_size,
                padding='valid',
                name='patch_embed_linear',
                **base_args
            )

        elif self.patch_embed_type == 'siglip':
            # Two-stage patch embedding. Not a SigLIP feature -- SigLIP is a
            # sigmoid contrastive loss and its tower uses a single-conv stem; see models/vision/vit_siglip/model.py's module docstring.
            return keras.Sequential([
                # Stage 1: Coarse-grained patching
                layers.Conv2D(
                    filters=self.embed_dim // 2,
                    kernel_size=self.patch_size // 2,
                    strides=self.patch_size // 2,
                    padding='valid',
                    name='patch_embed_conv1',
                    **base_args
                ),
                create_normalization_layer(
                    self.normalization_type,
                    name='patch_embed_norm1',
                    **self.norm_args
                ),
                layers.Activation('gelu', name='patch_embed_activation1'),
                # Stage 2: Refinement to final embedding dimension
                layers.Conv2D(
                    filters=self.embed_dim,
                    kernel_size=2,
                    strides=2,
                    padding='valid',
                    name='patch_embed_conv2',
                    **base_args
                ),
            ], name='patch_embed_siglip')

        elif self.patch_embed_type == 'conv':
            # Multi-layer convolution patch embedding
            return keras.Sequential([
                layers.Conv2D(
                    filters=self.embed_dim // 4,
                    kernel_size=3,
                    strides=2,
                    padding='same',
                    name='patch_embed_conv1',
                    **base_args
                ),
                create_normalization_layer(
                    self.normalization_type,
                    name='patch_embed_norm1',
                    **self.norm_args
                ),
                layers.Activation(self.activation, name='patch_embed_act1'),
                layers.Conv2D(
                    filters=self.embed_dim // 2,
                    kernel_size=3,
                    strides=2,
                    padding='same',
                    name='patch_embed_conv2',
                    **base_args
                ),
                create_normalization_layer(
                    self.normalization_type,
                    name='patch_embed_norm2',
                    **self.norm_args
                ),
                layers.Activation(self.activation, name='patch_embed_act2'),
                layers.Conv2D(
                    filters=self.embed_dim,
                    kernel_size=self.patch_size // 4,
                    strides=self.patch_size // 4,
                    padding='valid',
                    name='patch_embed_conv3',
                    **base_args
                ),
            ], name='patch_embed_conv')

        else:  # hybrid
            # Hybrid CNN backbone + patch embedding (simplified)
            return keras.Sequential([
                # CNN backbone (simplified ResNet-like)
                layers.Conv2D(64, 7, strides=2, padding='same', name='hybrid_conv1', **base_args),
                create_normalization_layer(self.normalization_type, name='hybrid_norm1', **self.norm_args),
                layers.Activation(self.activation, name='hybrid_act1'),
                layers.MaxPooling2D(3, strides=2, padding='same', name='hybrid_pool1'),
                # Bottleneck
                layers.Conv2D(self.embed_dim // 2, 3, padding='same', name='hybrid_conv2', **base_args),
                create_normalization_layer(self.normalization_type, name='hybrid_norm2', **self.norm_args),
                layers.Activation(self.activation, name='hybrid_act2'),
                # Final patch embedding
                layers.Conv2D(
                    filters=self.embed_dim,
                    kernel_size=1,
                    strides=1,
                    padding='valid',
                    name='hybrid_patch_embed',
                    **base_args
                ),
            ], name='patch_embed_hybrid')

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the vision encoder and all sub-layers.

        :param input_shape: Shape ``(batch, height, width, channels)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If shape is invalid.
        """
        if self.built:
            return

        if len(input_shape) != 4:
            raise ValueError(
                f"Expected 4D input shape (batch, height, width, channels), "
                f"got {input_shape}"
            )

        # CLS token is created in __init__ as its shape is independent of input.

        # Build patch embedding layer
        self.patch_embed.build(input_shape)

        # Build positional embedding
        pos_input_shape = (None, self.seq_len, self.embed_dim)
        self.pos_embed.build(pos_input_shape)

        # Build transformer layers
        for layer in self.transformer_layers:
            layer.build(pos_input_shape)

        # Build final normalization if present
        if self.norm is not None:
            self.norm.build(pos_input_shape)

        # Build pooling layer
        self.pooling_layer.build(pos_input_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def _extend_mask_for_cls(
            self,
            attention_mask: Optional[keras.KerasTensor],
            batch_size: Any
    ) -> Optional[keras.KerasTensor]:
        """Splice an always-attend CLS entry onto a patch-level attention mask.

        Shared contract (2 call sites: ``_get_full_sequence_features`` and
        ``call``). Given the caller's ``(B, num_patches)`` keep-mask, returns a
        ``(B, 1 + num_patches)`` mask whose position 0 is 1 whenever
        ``use_cls_token`` is True; returns the argument unchanged when it is
        ``None`` or when there is no CLS token. Never raises; validation of the
        mask's rank happens once, at the public entry points.

        :param attention_mask: Patch-level keep-mask ``(B, num_patches)``, 1 = attend.
        :type attention_mask: Optional[keras.KerasTensor]
        :param batch_size: Dynamic batch size, from ``ops.shape(inputs)[0]``.
        :type batch_size: Any
        :return: CLS-extended mask, or the input unchanged.
        :rtype: Optional[keras.KerasTensor]
        """
        if attention_mask is None or not self.use_cls_token:
            return attention_mask

        # DECISION plan-2026-07-31T042809-ddc92265/D-009: splice a ones column
        # for the CLS token at position 0 -- an un-extended mask misaligns every downstream index by one (13 of 18 pooling strategies broke on this). See decisions.md.
        cls_mask = ops.ones((batch_size, 1), dtype=attention_mask.dtype)
        return ops.concatenate([cls_mask, attention_mask], axis=1)

    def _validate_attention_mask(self, attention_mask: Optional[keras.KerasTensor]) -> None:
        """Fail loudly on a mask that does not match the documented contract.

        :param attention_mask: Candidate mask.
        :type attention_mask: Optional[keras.KerasTensor]
        :raises ValueError: If the mask is not rank-2.
        """
        if attention_mask is None:
            return
        if len(attention_mask.shape) != 2:
            raise ValueError(
                f"attention_mask must be a rank-2 keep-mask of shape "
                f"(batch, num_patches); got shape {tuple(attention_mask.shape)}. "
                f"The mask is over patches - the CLS token is excluded and is "
                f"spliced in internally."
            )

    def _get_full_sequence_features(
            self,
            inputs: keras.KerasTensor,
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Internal helper to run the full forward pass and return sequence features.

        :param inputs: Image tensor ``(B, H, W, C)``.
        :type inputs: keras.KerasTensor
        :param attention_mask: Optional patch-level keep-mask ``(B, num_patches)``,
            1 = attend. CLS excluded; spliced in here when ``use_cls_token=True``.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Training mode flag.
        :type training: Optional[bool]
        :return: Sequence features ``(B, seq_len, embed_dim)``.
        :rtype: keras.KerasTensor
        """
        batch_size = ops.shape(inputs)[0]
        x = self.patch_embed(inputs, training=training)

        # Reshape to sequence format. Shape can vary by patch embedder.
        # Final shape should be (batch_size, num_patches, embed_dim)
        if len(x.shape) == 4:
            x = ops.reshape(x, [batch_size, -1, self.embed_dim])

        if self.use_cls_token:
            cls_tokens = ops.broadcast_to(self.cls_token, (batch_size, 1, self.embed_dim))
            x = ops.concatenate([cls_tokens, x], axis=1)

        attention_mask = self._extend_mask_for_cls(attention_mask, batch_size)

        x = self.pos_embed(x, training=training)

        for layer in self.transformer_layers:
            x = layer(x, attention_mask=attention_mask, training=training)

        if self.norm is not None:
            x = self.norm(x, training=training)

        return x

    def call(
            self,
            inputs: keras.KerasTensor,
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the vision encoder.

        :param inputs: Image tensor ``(B, H, W, C)``.
        :type inputs: keras.KerasTensor
        :param attention_mask: Optional keep-mask ``(B, num_patches)``, 1 = attend,
            0 = mask. The mask is over patches: the CLS token is excluded from
            it and is always kept (a ones column is spliced in internally when
            ``use_cls_token=True``). It gates every ``TransformerLayer`` in the
            stack as well as the final pooling.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Training mode flag.
        :type training: Optional[bool]
        :return: Output features (shape depends on ``output_mode``).
        :rtype: keras.KerasTensor
        :raises ValueError: If ``attention_mask`` is not rank-2.

        Every ``output_mode`` is ACCEPTED with a mask — no mode raises any more.
        ``'weighted'``, ``'top_k_mean'`` and ``'top_k_max'`` used to be REFUSED
        here because their pooled output was not isolated from a masked patch
        (F-24); that defect is fixed in ``layers/sequence_pooling/`` and the
        refusal is gone — see the
        ``# DECISION plan-2026-07-31T132403-b3f540cb/D-003`` note at the top of
        this module.

        Acceptance is NOT a blanket promise of isolation. Two modes are
        exempt BY DEFINITION and one is exempt BY INTENT:

        * ``'none'`` and ``'flatten'`` return the per-token sequence, so a
          masked token's own row is present by definition.
        * ``'cls'`` and ``'first'`` return the token at index 0 BY INTENT,
          mask or no mask. With ``use_cls_token=True`` index 0 is the CLS
          token, which is never masked (see :meth:`_extend_mask_for_cls`), so
          they isolate; with ``use_cls_token=False`` index 0 is patch 0, and
          masking patch 0 while asking for ``'cls'``/``'first'`` is a
          contradiction the layer answers literally rather than redirecting.
        * ``'last'`` and ``'middle'`` ARE mask-aware as of F-25's closure:
          ``'last'`` returns the last KEPT position and ``'middle'`` the
          middle of the KEPT positions, both derived from the mask rather than
          from the padded length. They no longer return a masked token for any
          mask that keeps at least one position — see ``SequencePooling``'s
          class docstring for the exhaustive-mask measurement that backs that
          quantifier and for its exact scope. Do NOT re-add a "leaks under a
          contiguous-prefix mask" caveat for ``'middle'``: that was true of the
          pre-F-25 code and is now measurably false.
        * Every NON-positional mode isolates.

        The per-mode semantics live ONCE, in
        :class:`~dl_techniques.layers.sequence_pooling.sequence_pooling.SequencePooling`'s
        class docstring; the measured through-this-layer sweep that backs the
        four bullets above lives ONCE, next to ``ISOLATING_OUTPUT_MODES`` in
        ``tests/test_layers/test_transformers/test_vision_encoder.py``.
        """
        self._validate_attention_mask(attention_mask)

        x = self._get_full_sequence_features(
            inputs, attention_mask=attention_mask, training=training
        )

        pooling_mask = self._extend_mask_for_cls(attention_mask, ops.shape(inputs)[0])
        output = self.pooling_layer(x, mask=pooling_mask, training=training)

        return output

    def get_cls_features(
            self,
            inputs: keras.KerasTensor,
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Extract CLS token features for classification.

        :param inputs: Image tensor.
        :type inputs: keras.KerasTensor
        :param attention_mask: Optional patch-level keep-mask ``(B, num_patches)``;
            see :meth:`call`.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Training mode flag.
        :type training: Optional[bool]
        :return: CLS features ``(B, embed_dim)``.
        :rtype: keras.KerasTensor
        :raises ValueError: If ``use_cls_token=False`` or the mask is not rank-2.
        """
        if not self.use_cls_token:
            raise ValueError("CLS token is not available when use_cls_token=False")

        self._validate_attention_mask(attention_mask)
        features = self._get_full_sequence_features(
            inputs, attention_mask=attention_mask, training=training
        )
        return features[:, 0, :]

    def get_patch_features(
            self,
            inputs: keras.KerasTensor,
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Extract patch token features for dense prediction.

        :param inputs: Image tensor.
        :type inputs: keras.KerasTensor
        :param attention_mask: Optional patch-level keep-mask ``(B, num_patches)``;
            see :meth:`call`.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Training mode flag.
        :type training: Optional[bool]
        :return: Patch features ``(B, num_patches, embed_dim)``.
        :rtype: keras.KerasTensor
        :raises ValueError: If ``attention_mask`` is not rank-2.
        """
        self._validate_attention_mask(attention_mask)
        features = self._get_full_sequence_features(
            inputs, attention_mask=attention_mask, training=training
        )
        if self.use_cls_token:
            return features[:, 1:, :]
        else:
            return features

    def get_spatial_features(
            self,
            inputs: keras.KerasTensor,
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Get spatial features reshaped for dense prediction.

        :param inputs: Image tensor.
        :type inputs: keras.KerasTensor
        :param attention_mask: Optional patch-level keep-mask ``(B, num_patches)``;
            see :meth:`call`.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Training mode flag.
        :type training: Optional[bool]
        :return: Spatial features ``(B, patch_H, patch_W, embed_dim)``.
        :rtype: keras.KerasTensor
        :raises ValueError: If ``attention_mask`` is not rank-2.
        """
        patch_features = self.get_patch_features(
            inputs, attention_mask=attention_mask, training=training
        )
        batch_size = ops.shape(patch_features)[0]

        patches_h = self.img_size // self.patch_size
        patches_w = self.img_size // self.patch_size

        return ops.reshape(
            patch_features,
            [batch_size, patches_h, patches_w, self.embed_dim]
        )

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape.

        :param input_shape: Input shape ``(B, H, W, C)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape (depends on ``output_mode``).
        :rtype: Tuple[Optional[int], ...]
        """
        batch_size = input_shape[0]

        # Create dummy sequence shape for pooling layer
        sequence_shape = (batch_size, self.seq_len, self.embed_dim)
        return self.pooling_layer.compute_output_shape(sequence_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary for serialization.

        :return: Dictionary containing all constructor parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'img_size': self.img_size,
            'patch_size': self.patch_size,
            'embed_dim': self.embed_dim,
            'depth': self.depth,
            'num_heads': self.num_heads,
            'mlp_ratio': self.mlp_ratio,
            'patch_embed_type': self.patch_embed_type,
            'attention_type': self.attention_type,
            'normalization_type': self.normalization_type,
            'normalization_position': self.normalization_position,
            'ffn_type': self.ffn_type,
            'use_cls_token': self.use_cls_token,
            'output_mode': self.output_mode,
            'dropout_rate': self.dropout_rate,
            'attention_dropout_rate': self.attention_dropout_rate,
            'pos_dropout_rate': self.pos_dropout_rate,
            'stochastic_depth_rate': self.stochastic_depth_rate,
            'activation': keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'attention_args': self.attention_args,
            'norm_args': self.norm_args,
            'ffn_args': self.ffn_args,
            'patch_embed_args': self.patch_embed_args,
        })
        return config


# ---------------------------------------------------------------------
# Factory Functions for Convenient Encoder Creation
# ---------------------------------------------------------------------


def create_vision_encoder(
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        patch_embed_type: PatchEmbedType = 'linear',
        attention_type: AttentionType = 'multi_head',
        normalization_type: NormalizationType = 'layer_norm',
        normalization_position: NormalizationPositionType = 'post',
        ffn_type: FFNType = 'mlp',
        use_cls_token: bool = True,
        output_mode: PoolingStrategy = 'cls',
        dropout_rate: float = 0.0,
        **kwargs: Any
) -> VisionEncoder:
    """
    Factory function to create a VisionEncoder with validated parameters.

    This function provides parameter validation and sensible defaults for creating
    vision_heads encoders with different architectural configurations. It supports all
    major vision_heads transformer variants through configurable components.

    :param img_size: Input image size. Must be divisible by patch_size.
    :type img_size: int
    :param patch_size: Size of image patches.
    :type patch_size: int
    :param embed_dim: Embedding dimension.
    :type embed_dim: int
    :param depth: Number of transformer layers.
    :type depth: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param mlp_ratio: MLP expansion ratio.
    :type mlp_ratio: float
    :param patch_embed_type: Type of patch embedding strategy.
    :type patch_embed_type: str
    :param attention_type: Type of attention mechanism.
    :type attention_type: str
    :param normalization_type: Type of normalization.
    :type normalization_type: str
    :param normalization_position: Position of normalization layers.
    :type normalization_position: str
    :param ffn_type: Type of feed-forward network.
    :type ffn_type: str
    :param use_cls_token: Whether to use CLS token.
    :type use_cls_token: bool
    :param output_mode: Output pooling mode.
    :type output_mode: str
    :param dropout_rate: General dropout rate.
    :type dropout_rate: float
    :param kwargs: Additional arguments for VisionEncoder constructor.
    :return: Configured VisionEncoder instance.
    :rtype: VisionEncoder
    :raises ValueError: If any parameter validation fails.
    """
    # Validate basic parameters
    if img_size <= 0 or patch_size <= 0:
        raise ValueError(f"img_size and patch_size must be positive, got {img_size}, {patch_size}")

    if img_size % patch_size != 0:
        raise ValueError(f"img_size ({img_size}) must be divisible by patch_size ({patch_size})")

    if embed_dim <= 0 or depth <= 0 or num_heads <= 0:
        raise ValueError("embed_dim, depth, and num_heads must be positive")

    if embed_dim % num_heads != 0:
        raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

    return VisionEncoder(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        patch_embed_type=patch_embed_type,
        attention_type=attention_type,
        normalization_type=normalization_type,
        normalization_position=normalization_position,
        ffn_type=ffn_type,
        use_cls_token=use_cls_token,
        output_mode=output_mode,
        dropout_rate=dropout_rate,
        **kwargs
    )

# ---------------------------------------------------------------------

def create_vit_encoder(
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        **kwargs: Any
) -> VisionEncoder:
    """Create standard ViT encoder configuration."""
    return create_vision_encoder(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        patch_embed_type='linear',
        attention_type='multi_head',
        normalization_type='layer_norm',
        normalization_position='post',
        ffn_type='mlp',
        **kwargs
    )

# ---------------------------------------------------------------------

def create_siglip_encoder(
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        **kwargs: Any
) -> VisionEncoder:
    """Create SigLIP-style encoder configuration."""
    return create_vision_encoder(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        patch_embed_type='siglip',
        attention_type='multi_head',
        normalization_type='layer_norm',
        normalization_position='post',
        ffn_type='mlp',
        **kwargs
    )

# ---------------------------------------------------------------------