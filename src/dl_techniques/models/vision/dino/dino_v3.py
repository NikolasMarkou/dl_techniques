"""
Pre-norm vision transformer in the DINO family, with optional rotary position
embeddings.

The DINOv3 paper answers a problem where self-distillation, run long enough,
degrades dense features (segmentation, depth) even as global features keep
improving: Gram anchoring, an extra loss pulling the student's patch-feature
Gram matrix toward an earlier frozen copy, plus a positional scheme that does
not bake in one resolution. This file implements only the second half.
`positional_embedding_type='rope'` replaces the learned absolute table with
rotary embeddings applied to Q and K inside the attention operator, so the
attention score depends on the difference of two positions rather than their
absolute values, generalizing across sequence lengths a fixed table cannot
represent. Gram anchoring is not implemented: no frozen Gram teacher and no
Gram-matrix loss term exist in this repository.

RoPE is reached through the registered `group_query` attention with
`num_kv_heads == num_heads`, which reduces grouped-query attention to
ordinary multi-head attention that rotates Q and K after projection, rather
than rotating the token stream before it (which would destroy the
relative-position property RoPE is defined by). Under `'rope'` the learned
absolute table is omitted rather than stacked on the rotation, since two
position signals would be redundant and the table alone would break
permutation equivariance. `rope_percentage=0.0` is legal and leaves the
model with no positional information at all; it exists as the control arm of
the RoPE-liveness test, not as a training configuration. A checkpoint is not
portable between the two positional modes, since they instantiate different
attention classes.

Everything else is a conventional pre-norm ViT: patch embedding, a learnable
[CLS] token, `depth` transformer blocks with linearly increasing stochastic
depth, a final normalization, and the [CLS] row as the feature vector,
optionally followed by a classifier.

Several other DINOv3 mechanisms are absent. The RoPE here is 1-D over the
flattened token sequence (position = token index); the paper uses a 2-D
axial formulation over patch (row, column) coordinates with random
coordinate jittering during training. Sinkhorn-Knopp centering is not
implemented (`dl_techniques.losses.dino_loss` offers EMA centering only).
Register tokens are not implemented in this model;
`dino_v2.DINOv2VisionTransformer` has them. High-resolution adaptation and
distillation from a large pretrained teacher are not implemented, and no
pretrained weights are shipped: `pretrained=True` raises
`NotImplementedError` rather than returning a randomly initialized model.

`patch_size=None` is the only place in the DINO trio where it resolves to
something other than a constant: `giant` carries `(14, 14)` and
`stochastic_depth_rate=0.4` while every other variant carries `(16, 16)`.
`get_last_selfattention` raises `NotImplementedError` rather than returning
a zero tensor a caller cannot distinguish from a broken model, since under
`'learned'` the multi-head attention classes accept no
`return_attention_scores` at all, while under `'rope'`
`GroupedQueryAttention` returns a correct probability map but
`TransformerLayer.call` never forwards the flag.

References:
    - Siméoni et al., 2025. DINOv3. (arXiv preprint; Gram anchoring and 2-D axial
      RoPE, neither implemented here)
    - Caron et al., 2021. Emerging Properties in Self-Supervised Vision
      Transformers. (https://arxiv.org/abs/2104.14294)
    - Oquab et al., 2023. DINOv2: Learning Robust Visual Features without
      Supervision. (https://arxiv.org/abs/2304.07193)
    - Su et al., 2021. RoFormer: Enhanced Transformer with Rotary Position
      Embedding. (https://arxiv.org/abs/2104.09864)
    - Dosovitskiy et al., 2020. An Image is Worth 16x16 Words: Transformers for
      Image Recognition at Scale. (https://arxiv.org/abs/2010.11929)
    - Huang et al., 2016. Deep Networks with Stochastic Depth.
      (https://arxiv.org/abs/1603.09382)
"""

import keras
from keras import layers, initializers, regularizers
from typing import Optional, Union, Tuple, Dict, Any, Callable, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.initializers import clone_initializer
from dl_techniques.utils.logger import logger
from dl_techniques.utils.drop_path import linear_drop_path_rates
from dl_techniques.layers.transformers import TransformerLayer
from dl_techniques.layers.embedding.patch_embedding import PatchEmbedding2D
from dl_techniques.layers.embedding.positional_embedding import PositionalEmbedding
from dl_techniques.layers.embedding.class_token import ClassTokenPrepend
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.models.vision.dino.common import reject_input_shape
from dl_techniques.models.vision.dino.reference_init import DINO_KERNEL_INITIALIZER
from dl_techniques.utils.activation_serialization import (
    serialize_activation,
    deserialize_activation,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.dino.dino_v3")
class DINOv3(keras.Model):
    """
    DINOv3 Vision Transformer Model Implementation.

    A pre-normalization Vision Transformer backbone, usable for classification and
    as the trunk of DINO-style self-supervised training. Pre-norm is what the DINO
    line relies on for self-distillation stability.

    The model consists of:
    - A patch embedding layer to convert images into sequences of tokens.
    - A learnable [CLS] token for global image representation.
    - Positional information, selected by ``positional_embedding_type``:
      either a learned absolute embedding table added to the token stream
      (``"learned"``), or 1-D rotary position embeddings applied to Q and K
      inside every attention operator (``"rope"``). The two are mutually
      exclusive — under ``"rope"`` no learned table is created at all.
    - A stack of pre-normalized Transformer encoder layers.
    - A final normalization layer.
    - An optional classification head.

    Implemented vs. not implemented (DINOv3 mechanisms):
        RoPE is implemented (1-D, over the flattened token sequence).
        **Gram anchoring is NOT implemented** — it requires a third frozen "Gram
        teacher" network and is an explicit non-goal here. **Sinkhorn-Knopp
        centering is NOT implemented.** **2-D axial RoPE with coordinate jittering
        is NOT implemented** — the rotation here is 1-D over token index.
        **Register tokens are NOT implemented** in this model. See the module
        docstring for the full list.

    Args:
        image_size: Integer, or tuple of integers (height, width), for the input
            image. An integer ``s`` is normalized to ``(s, s)``.
            Defaults to (224, 224).
        patch_size: Integer, or tuple of integers (height, width), for the image
            patches. An integer ``p`` is normalized to ``(p, p)``.
            Defaults to (16, 16).
        num_classes: Number of output classes for the classification head. If 0,
            no head is added. Defaults to 1000.
        embed_dim: The dimensionality of the token embeddings. Defaults to 768.
        depth: The number of transformer encoder layers. Defaults to 12.
        num_heads: The number of attention heads in each transformer layer.
            Defaults to 12.
        mlp_ratio: Ratio to determine the hidden dimension of the FFN in
            transformer layers (hidden_dim = embed_dim * mlp_ratio). Defaults to 4.0.
        qkv_bias: If True, add a learnable bias to the query, key, and value
            projections. Defaults to True.
        dropout_rate: Dropout rate for the embedding and FFN layers.
            Defaults to 0.0.
        attention_dropout_rate: Dropout rate for the attention weights.
            Defaults to 0.0.
        stochastic_depth_rate: Maximum drop rate for stochastic depth, which
            linearly increases across layers. Defaults to 0.0.
        normalization_type: The type of normalization to use ('layer_norm',
            'rms_norm'). Defaults to 'layer_norm'.
        positional_embedding_type: How positional information enters the model.
            ``'learned'`` (default) adds a learned absolute embedding table to the
            token stream. ``'rope'`` instead applies rotary position embeddings to
            Q and K inside each attention operator and creates **no** learned
            table. Defaults to 'learned'.
        rope_theta: Base frequency for the rotary embeddings. Only read when
            ``positional_embedding_type='rope'``. Defaults to 10000.0.
        rope_percentage: Fraction of each head's dimensions the rotation is
            applied to. Only read when ``positional_embedding_type='rope'``.
            **0.0 disables the rotation entirely**, which — since no learned table
            is created under ``'rope'`` — leaves the model with NO positional
            information at all (a permutation-equivariant bag of patches). That
            configuration is legal because it is the control arm the RoPE-liveness
            test needs; it is not a useful training configuration.
            Defaults to 1.0.
        activation: Activation function for the FFN layers. Defaults to 'gelu'.
        kernel_initializer: Initializer for kernel weights. Defaults to
            ``TruncatedNormal(stddev=0.02)``, DINO's published
            ``trunc_normal_(std=.02)``
            (https://github.com/facebookresearch/dinov3/blob/main/dinov3/models/vision_transformer.py).
        bias_initializer: Initializer for bias weights. Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer for kernel weights.
        bias_regularizer: Optional regularizer for bias weights.
        include_top: If True, include the final classification head. If False,
            the model outputs features from the transformer. Defaults to True.
        **kwargs: Additional arguments for the `keras.Model` base class.

    Input shape:
        A 4D tensor of shape `(batch_size, height, width, channels)`, where
        height and width must match `image_size`.

    Output shape:
        - If `include_top=True`: A 2D tensor of shape `(batch_size, num_classes)`.
        - If `include_top=False`: A 2D tensor of shape `(batch_size, embed_dim)`
          representing the [CLS] token features.

    Raises:
        ValueError: If model parameters are invalid or incompatible.
    """

    MODEL_VARIANTS = {
        "tiny": {
            "embed_dim": 192, "depth": 12, "num_heads": 3, "mlp_ratio": 4.0,
            "patch_size": (16, 16)
        },
        "small": {
            "embed_dim": 384, "depth": 12, "num_heads": 6, "mlp_ratio": 4.0,
            "patch_size": (16, 16)
        },
        "base": {
            "embed_dim": 768, "depth": 12, "num_heads": 12, "mlp_ratio": 4.0,
            "patch_size": (16, 16)
        },
        "large": {
            "embed_dim": 1024, "depth": 24, "num_heads": 16, "mlp_ratio": 4.0,
            "patch_size": (16, 16)
        },
        "giant": {
            "embed_dim": 1536, "depth": 40, "num_heads": 24, "mlp_ratio": 4.0,
            "patch_size": (14, 14), "stochastic_depth_rate": 0.4
        }
    }

    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]] = (224, 224),
        patch_size: Union[int, Tuple[int, int]] = (16, 16),
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        stochastic_depth_rate: float = 0.0,
        normalization_type: Literal['layer_norm', 'rms_norm'] = 'layer_norm',
        positional_embedding_type: Literal['learned', 'rope'] = 'learned',
        rope_theta: float = 10000.0,
        rope_percentage: float = 1.0,
        activation: Union[str, Callable] = 'gelu',
        # DECISION plan-2026-08-23T091307-9a110062/D-504: default to DINO_KERNEL_INITIALIZER, never 'glorot_uniform' or the bare string "truncated_normal".
        # The bare string resolves to Keras' stddev=0.05, 2.5x wider than DINOv3's published std=.02. See decisions.md.
        kernel_initializer: Union[str, Dict[str, Any], initializers.Initializer] = DINO_KERNEL_INITIALIZER,
        bias_initializer: Union[str, initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        bias_regularizer: Optional[regularizers.Regularizer] = None,
        include_top: bool = True,
        **kwargs: Any
    ) -> None:
        # Normalize int-or-tuple spellings BEFORE any subscript. Passing
        # `image_size=224` used to crash on `image_size[0]` with a bare TypeError
        # instead of building a 224x224 model, unlike v1/v2 which both accept an int.
        image_size = (
            tuple(image_size) if isinstance(image_size, (tuple, list))
            else (image_size, image_size)
        )
        patch_size = (
            tuple(patch_size) if isinstance(patch_size, (tuple, list))
            else (patch_size, patch_size)
        )

        # Input validation
        if image_size[0] % patch_size[0] != 0 or image_size[1] % patch_size[1] != 0:
            raise ValueError(f"image_size {image_size} must be divisible by patch_size {patch_size}")
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        if positional_embedding_type not in ('learned', 'rope'):
            raise ValueError(
                f"positional_embedding_type must be 'learned' or 'rope', "
                f"got '{positional_embedding_type}'"
            )
        if rope_theta <= 0.0:
            raise ValueError(f"rope_theta must be positive, got {rope_theta}")
        if not 0.0 <= rope_percentage <= 1.0:
            raise ValueError(f"rope_percentage must be in [0, 1], got {rope_percentage}")

        # DECISION plan_2026-06-15_39a31d4a/D-001: call `super().__init__(inputs=, outputs=)` exactly once, at the end of `__init__`.
        # A bare `super().__init__(**kwargs)` here double-initializes the Functional model. See decisions.md.

        # Store configuration
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.stochastic_depth_rate = stochastic_depth_rate
        self.normalization_type = normalization_type
        self.positional_embedding_type = positional_embedding_type
        self.rope_theta = rope_theta
        self.rope_percentage = rope_percentage
        self.activation = deserialize_activation(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.include_top = include_top

        # Compute derived values
        self.num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        self.sequence_length = self.num_patches + 1

        # Build the model using the functional API pattern
        inputs = keras.Input(shape=(*image_size, 3), name="input_image")
        outputs = self._build_model(inputs)

        # Finalize the Model
        # DECISION plan-2026-08-19T163559-499b6f0e/D-082: `name` stays a default via `setdefault`, never a hard-coded literal.
        # `from_config` passes `name` through `**kwargs`; a hard-coded `name=` beside it raises a duplicate-keyword TypeError. See decisions.md.
        kwargs.setdefault("name", "DINOv3")
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

        logger.info(
            f"Created DINOv3 model with {depth} layers, {embed_dim} embedding dim for "
            f"input shape {image_size}"
        )

    def _build_model(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Constructs the model architecture."""
        # 1. Patch Embedding
        x = self._build_patch_embedding(inputs)

        # 2. Add CLS Token and Positional Embedding
        x = self._build_token_processing(x)

        # 3. Transformer Encoder Layers
        x = self._build_encoder(x)

        # 4. Final Processing and Head
        x = self._build_head(x)

        return x

    def _build_patch_embedding(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Creates the patch embedding layer."""
        # DECISION plan-2026-08-23T091307-9a110062/D-540: give every consumer its own `clone_initializer(...)` copy, never `self.kernel_initializer` directly.
        # A shared seedless initializer instance replays its draw, so every same-shape kernel it reaches is bit-identical. See decisions.md.
        self.patch_embed = PatchEmbedding2D(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='patch_embedding'
        )
        return self.patch_embed(inputs)

    def _build_token_processing(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Adds the [CLS] token and, for the 'learned' mode, positional embeddings."""
        # DECISION plan_2026-06-15_39a31d4a/D-001: own the CLS token via ClassTokenPrepend, never an inline `self.add_weight` here.
        # This runs inside `_build_model`, before `super().__init__`; an inline add_weight would fire too early and crash. See decisions.md.
        self.cls_token_layer = ClassTokenPrepend(name="cls_token")
        x = self.cls_token_layer(x)

        # DECISION plan-2026-08-01T105809-dc0c402e/D-015: omit the learned absolute table under 'rope', never stack it on top of the rotation.
        # Both together give two redundant position signals, and the table alone breaks permutation equivariance. See decisions.md.
        if self.positional_embedding_type == 'rope':
            self.pos_embed = None
            # PositionalEmbedding also owned the post-embedding dropout; preserve it.
            if self.dropout_rate > 0.0:
                self.embed_dropout = layers.Dropout(
                    self.dropout_rate, name="embedding_dropout"
                )
                x = self.embed_dropout(x)
            else:
                self.embed_dropout = None
            return x

        # Add positional embedding using the shared layer
        self.embed_dropout = None
        self.pos_embed = PositionalEmbedding(
            max_seq_len=self.sequence_length,
            dim=self.embed_dim,
            dropout_rate=self.dropout_rate,
            name="positional_embedding"
        )
        return self.pos_embed(x)

    def _attention_spec(self) -> Tuple[str, Dict[str, Any]]:
        """The attention type + factory args every encoder block is built with.

        Returns ``(attention_type, attention_args)``. Under
        ``positional_embedding_type='rope'`` this selects the rope-capable
        ``group_query`` attention with ``num_kv_heads == num_heads``; otherwise the
        plain ``multi_head`` attention, byte-identically to the pre-RoPE behaviour.
        """
        # DECISION plan-2026-08-01T105809-dc0c402e/D-015: reach RoPE through registered `group_query` attention, never by rotating the token stream directly.
        # RoPE must rotate Q and K after projection; `num_kv_heads == num_heads` makes GQA reduce to plain MHA. Do not simplify to `multi_head` plus a rope kwarg. See decisions.md.
        if self.positional_embedding_type == 'rope':
            return 'group_query', {
                'num_kv_heads': self.num_heads,
                'rope_theta': self.rope_theta,
                'rope_percentage': self.rope_percentage,
                'max_seq_len': self.sequence_length,
                'use_bias': self.qkv_bias,
            }
        return 'multi_head', {'use_bias': self.qkv_bias}

    def _build_encoder(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Creates the stack of transformer encoder layers."""
        self.encoder_layers = []
        # DECISION plan-2026-08-11T165740-53dac34a/D-004: use the shared `linear_drop_path_rates` helper, never a hand-rolled `ops.linspace` ramp.
        # It returns plain Python floats; a keras tensor has no `.item()`. See decisions.md.
        dpr = linear_drop_path_rates(self.depth, self.stochastic_depth_rate)

        attention_type, attention_args = self._attention_spec()

        for i in range(self.depth):
            encoder_layer = TransformerLayer(
                hidden_size=self.embed_dim,
                num_heads=self.num_heads,
                intermediate_size=int(self.embed_dim * self.mlp_ratio),
                attention_type=attention_type,
                attention_args=dict(attention_args),
                normalization_type=self.normalization_type,
                normalization_position='pre',  # DINO uses pre-norm
                ffn_type='mlp',
                activation=self.activation,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                use_stochastic_depth=dpr[i] > 0.0,
                stochastic_depth_rate=dpr[i],
                use_bias=True,
                kernel_initializer=clone_initializer(self.kernel_initializer),
                bias_initializer=clone_initializer(self.bias_initializer),
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name=f'encoder_layer_{i}'
            )
            x = encoder_layer(x)
            self.encoder_layers.append(encoder_layer)
        return x

    def _build_head(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Creates the final normalization and classification head."""
        # Final normalization using the shared factory
        self.norm = create_normalization_layer(
            self.normalization_type,
            name='final_norm'
        )
        x = self.norm(x)

        # Extract [CLS] token for classification
        features = x[:, 0]

        # Add classification head if requested
        if self.include_top:
            if self.num_classes > 0:
                # DECISION plan-2026-08-23T091307-9a110062/D-504: use DINO_KERNEL_INITIALIZER here too, never the bare string.
                # The bare string is Keras' stddev=0.05, not DINO's 0.02. See decisions.md.
                self.classifier = layers.Dense(
                    units=self.num_classes,
                    kernel_initializer=initializers.get(DINO_KERNEL_INITIALIZER),
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    name='classifier'
                )
                outputs = self.classifier(features)
            else:
                # If include_top is True but num_classes is 0, return features
                outputs = features
        else:
            # If not including top, return features
            outputs = features

        return outputs

    def get_last_selfattention(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """
        Not implemented. Raises ``NotImplementedError``.

        DINO's attention-map visualization needs the last block's attention
        PROBABILITIES. This model composes ``TransformerLayer``, whose ``call()``
        signature is ``(inputs, attention_mask, layer_idx, training)`` and which
        returns only the block output — it has no way to surface its attention
        sub-layer's probabilities, under EITHER
        ``positional_embedding_type``. MEASURED on keras 3.8.0.

        The gap differs per path, and both are named here rather than collapsed:

        - ``'learned'`` -> ``multi_head`` attention: ``MultiHeadAttention.call`` and
          the ``MultiHeadCrossAttention.call`` it delegates to accept no
          ``return_attention_scores`` at all, and cache no probabilities.
        - ``'rope'`` -> ``group_query`` attention: ``GroupedQueryAttention.call``
          DOES accept ``return_attention_weights`` and returns a correct
          ``(batch, heads, seq, seq)`` map whose rows sum to 1 — but
          ``TransformerLayer.call`` does not forward that flag, so the capability
          is unreachable without reaching into block internals and re-implementing
          the block's pre-norm ordering in this model.

        Implementing this truthfully means adding the flag through
        ``TransformerLayer`` (and, for the learned path, through
        ``MultiHeadAttention`` / ``MultiHeadCrossAttention``) — a change to shared
        layers used across the repository, out of scope for this model file.

        Args:
            inputs: A batch of images. Unused.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "DINOv3.get_last_selfattention() is not implemented. It needs the last "
            "block's attention probabilities, and TransformerLayer.call does not "
            "forward a return_attention_scores / return_attention_weights flag. "
            "For positional_embedding_type='learned' the multi_head attention path "
            "(MultiHeadAttention / MultiHeadCrossAttention) does not accept such a "
            "flag either; for 'rope' the group_query path (GroupedQueryAttention) "
            "does accept return_attention_weights, but TransformerLayer does not "
            "pass it through. Previously this method returned an all-zero tensor, "
            "which is indistinguishable from a broken model."
        )

    @classmethod
    def from_variant(
        cls,
        variant: str,
        image_size: Union[int, Tuple[int, int]] = (224, 224),
        num_classes: int = 1000,
        include_top: bool = True,
        **kwargs: Any
    ) -> "DINOv3":
        """
        Creates a DINOv3 model from a predefined variant.

        Args:
            variant: The model variant, one of "tiny", "small", "base", "large", "giant".
            image_size: The input image size; an int, or (height, width).
            num_classes: Number of output classes.
            include_top: Whether to include the classification head.
            **kwargs: Additional arguments to pass to the model constructor.

        Returns:
            A DINOv3 model instance.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )
        config = cls.MODEL_VARIANTS[variant].copy()
        config.update(kwargs)

        logger.info(f"Creating DINOv3-{variant.upper()} model with config: {config}")

        return cls(
            image_size=image_size,
            num_classes=num_classes,
            include_top=include_top,
            **config
        )

    def get_config(self) -> Dict[str, Any]:
        """Returns the model's configuration for serialization."""
        # DECISION plan-2026-08-19T163559-499b6f0e/D-082: call `super().get_config()` first, never a literal dict.
        # Without it, `name` and `trainable` reload at their defaults, so a frozen model comes back unfrozen. See decisions.md.
        config = super().get_config()
        config.update({
            'image_size': self.image_size,
            'patch_size': self.patch_size,
            'num_classes': self.num_classes,
            'embed_dim': self.embed_dim,
            'depth': self.depth,
            'num_heads': self.num_heads,
            'mlp_ratio': self.mlp_ratio,
            'qkv_bias': self.qkv_bias,
            'dropout_rate': self.dropout_rate,
            'attention_dropout_rate': self.attention_dropout_rate,
            'stochastic_depth_rate': self.stochastic_depth_rate,
            'normalization_type': self.normalization_type,
            'positional_embedding_type': self.positional_embedding_type,
            'rope_theta': self.rope_theta,
            'rope_percentage': self.rope_percentage,
            'activation': serialize_activation(self.activation),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'include_top': self.include_top,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DINOv3":
        """Creates a model from its configuration."""
        return cls(**config)


def create_dino_v3(
    variant: str = "base",
    *,
    image_size: Union[int, Tuple[int, int]] = (224, 224),
    patch_size: Optional[Union[int, Tuple[int, int]]] = None,
    num_classes: int = 1000,
    include_top: bool = True,
    positional_embedding_type: Literal['learned', 'rope'] = 'learned',
    rope_theta: float = 10000.0,
    rope_percentage: float = 1.0,
    pretrained: bool = False,
    **kwargs: Any
) -> DINOv3:
    """
    A factory function to create DINOv3 models.

    Signature note (converged surface): ``create_dino_v1``, ``create_dino_v2`` and
    ``create_dino_v3`` share ``(variant, *, image_size, patch_size, num_classes,
    include_top, **kwargs)``. ``patch_size`` used to be reachable only through
    ``**kwargs`` here; it is now a named parameter on all three. There is no
    ``input_shape`` spelling on any of them — the input shape is derived from
    ``image_size``.

    **Variant-defers precedence rule** (shared by all three factories):
    ``patch_size=None`` defers to the variant's own ``MODEL_VARIANTS`` entry; an
    EXPLICIT non-``None`` value ALWAYS wins over it. This matters here and only
    here: ``DINOv3.MODEL_VARIANTS['giant']`` carries ``patch_size=(14, 14)`` while
    every other variant carries ``(16, 16)``, so ``create_dino_v3('giant')`` gives
    a /14 model and ``create_dino_v3('giant', patch_size=16)`` gives a /16 one.
    ``giant`` likewise carries ``stochastic_depth_rate=0.4``, overridable the same
    way by passing it through ``**kwargs``.

    Args:
        variant: Model variant ("tiny", "small", "base", "large", "giant").
        image_size: Input image size; an int, or (height, width).
        patch_size: Patch size; an int, or (height, width). ``None`` defers to the
            variant ((14, 14) for 'giant', (16, 16) otherwise).
        num_classes: Number of output classes.
        include_top: Whether to include the final classification layer.
        positional_embedding_type: ``'learned'`` (absolute table) or ``'rope'``
            (1-D rotary, applied inside a ``group_query`` attention). A checkpoint
            is NOT portable between the two — they run different attention classes.
        rope_theta: RoPE base frequency. Ignored unless
            ``positional_embedding_type='rope'``.
        rope_percentage: Fraction of each head's dimensions that are rotated.
            Ignored unless ``positional_embedding_type='rope'``. ``0.0`` is legal
            but leaves the model with NO positional information at all.
        pretrained: Must be False. `True` raises `NotImplementedError` — NO
            pretrained DINOv3 weights are shipped with this repository.
        **kwargs: Additional arguments for the model constructor, e.g.
            ``stochastic_depth_rate``, ``normalization_type``.

    Returns:
        A DINOv3 model instance.

    Raises:
        TypeError: If ``input_shape`` is passed — use ``image_size`` instead.
        NotImplementedError: If ``pretrained=True`` (no checkpoints shipped).
    """
    reject_input_shape(kwargs, "create_dino_v3")

    # DECISION plan-2026-08-14T233721-d4f9beb2/D-069: raise, do not warn-and-continue.
    if pretrained:
        raise NotImplementedError(
            f"No pretrained DINOv3 weights are distributed with dl_techniques "
            f"(requested variant '{variant}'). Build the architecture with "
            f"pretrained=False and warm-start from a local checkpoint instead: "
            f"model = create_dino_v3('{variant}', ...); "
            f"model.load_weights('/path/to/weights.keras'). Prefer "
            f"dl_techniques.utils.weight_transfer.load_weights_or_raise(model, "
            f"path), which raises when a load changes ZERO variables -- raw "
            f"load_weights is silent about a checkpoint that matches nothing."
        )

    # DECISION plan-2026-08-01T105809-dc0c402e/D-017: `patch_size=None` defers to the variant; never give this a concrete default like `= 16`.
    # A concrete default would always override the variant's and silently turn `create_dino_v3('giant')` from /14 into /16. See decisions.md.
    if patch_size is not None:
        kwargs['patch_size'] = patch_size

    return DINOv3.from_variant(
        variant=variant,
        image_size=image_size,
        num_classes=num_classes,
        include_top=include_top,
        positional_embedding_type=positional_embedding_type,
        rope_theta=rope_theta,
        rope_percentage=rope_percentage,
        **kwargs
    )
