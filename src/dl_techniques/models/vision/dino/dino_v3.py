"""
Pre-norm vision transformer in the DINO family, with optional rotary position
embeddings.

This file defines the ``DINOv3`` model and the ``create_dino_v3`` factory: patch
embedding, a learnable [CLS] token, ``depth`` pre-norm transformer blocks with a
linearly increasing stochastic-depth rate, a final normalization, and the [CLS]
row as the feature vector, optionally followed by a classifier.
``positional_embedding_type='rope'`` replaces the learned absolute position
table with rotary embeddings applied to Q and K inside the registered
``group_query`` attention, run with ``num_kv_heads == num_heads`` so it reduces
to ordinary multi-head attention. Under ``'rope'`` the table is omitted rather
than added on top of the rotation.

The rotation is 1-D over the flattened token sequence. Gram anchoring, 2-D axial
RoPE with coordinate jittering, Sinkhorn-Knopp centering and register tokens are
not implemented. No pretrained weights ship with this repository, so
``pretrained=True`` raises ``NotImplementedError``. A checkpoint is not portable
between the two positional modes, which instantiate different attention classes,
and ``rope_percentage=0.0`` leaves the model with no positional information at
all.

References:
    - Siméoni et al., 2025. DINOv3. (arXiv preprint)
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
    """Build a pre-norm DINO vision transformer with a selectable position scheme.

    The model is assembled with the functional API at construction time. Its one
    structural choice is ``positional_embedding_type``: ``'learned'`` adds an
    absolute embedding table to the token stream, while ``'rope'`` creates no
    table and instead rotates Q and K inside every attention operator. The two
    modes run different attention classes, so weights do not transfer between
    them.

    Architecture:

    .. code-block:: text

                     input_image [B, H, W, 3]
                                 ▼
                   ┌───────────────────────────┐
                   │ PatchEmbedding2D          │
                   └─────────────┬─────────────┘
                         [B, N, embed_dim]
                                 ▼
                   ┌───────────────────────────┐
                   │ ClassTokenPrepend         │
                   └─────────────┬─────────────┘
                        [B, N+1, embed_dim]
                                 ▼
                   ┌─────────────┴─────────────┐
                   ▼                           ▼
               'learned'                     'rope'
    ┌─────────────────────┐     ┌─────────────────────┐
    │ PositionalEmbedding │     │ Dropout             │
    │  table + dropout    │     │  only if dropout>0  │
    └──────────┬──────────┘     └──────────┬──────────┘
               └─────────────┬─────────────┘
                             ▼
                   ┌───────────────────────────┐
                   │ TransformerLayer x depth  │
                   │  pre-norm, drop-path ramp │
                   └─────────────┬─────────────┘
                                 ▼
                   ┌───────────────────────────┐
                   │ final_norm                │
                   └─────────────┬─────────────┘
                                 ▼
                   ┌───────────────────────────┐
                   │ take [CLS] row x[:, 0]    │
                   └─────────────┬─────────────┘
                      features [B, embed_dim]
                                 │
                   ┌─────────────┴─────────────┐
                   ▼                           ▼
    include_top, classes>0                 otherwise
    ┌─────────────────────┐                │
    │ Dense(num_classes)  │                │
    └──────────┬──────────┘                │
                   ▼                           ▼
           [B, num_classes]             [B, embed_dim]

    Encoder block (pre-norm, two residual halves):

    .. code-block:: text

                      x [B, N+1, D]
                            │
                ┌───────────┤
                │           ▼
                │   ┌───────────────┐
                │   │ norm          │
                │   └───────┬───────┘
                │           ▼
                │   ┌───────────────┐
                │   │ attention     │
                │   │  q, k rotated │
                │   │  ('rope' only)│
                │   └───────┬───────┘
                │           ▼
                │   ┌───────────────┐
                │   │ drop path     │
                │   │  (optional)   │
                │   └───────┬───────┘
                │           ▼
                │   ┌───────────────┐
                └──►│ add           │
                    └───────┬───────┘
                            ▼
                ┌───────────┤
                │           ▼
                │   ┌───────────────┐
                │   │ norm          │
                │   └───────┬───────┘
                │           ▼
                │   ┌───────────────┐
                │   │ mlp           │
                │   └───────┬───────┘
                │           ▼
                │   ┌───────────────┐
                │   │ drop path     │
                │   │  (optional)   │
                │   └───────┬───────┘
                │           ▼
                │   ┌───────────────┐
                └──►│ add           │
                    └───────┬───────┘
                            ▼
                      y [B, N+1, D]

    Variants:

    .. code-block:: text

        variant  embed_dim  depth  heads  mlp  patch  drop path
        ───────  ─────────  ─────  ─────  ───  ─────  ─────────
        tiny           192     12      3  4.0  16x16  (default)
        small          384     12      6  4.0  16x16  (default)
        base           768     12     12  4.0  16x16  (default)
        large         1024     24     16  4.0  16x16  (default)
        giant         1536     40     24  4.0  14x14        0.4

    (default) means the variant sets no value and ``stochastic_depth_rate``
    keeps whatever the caller passed.

    :param image_size: Input image size as ``(height, width)``; an integer ``s``
        becomes ``(s, s)``. Defaults to ``(224, 224)``.
    :type image_size: int or tuple
    :param patch_size: Patch size as ``(height, width)``; an integer ``p``
        becomes ``(p, p)``. Defaults to ``(16, 16)``.
    :type patch_size: int or tuple
    :param num_classes: Number of classifier outputs. ``0`` adds no head.
        Defaults to 1000.
    :type num_classes: int
    :param embed_dim: Token embedding width. Defaults to 768.
    :type embed_dim: int
    :param depth: Number of transformer blocks. Defaults to 12.
    :type depth: int
    :param num_heads: Attention heads per block. Defaults to 12.
    :type num_heads: int
    :param mlp_ratio: FFN hidden width as a multiple of ``embed_dim``. Defaults
        to 4.0.
    :type mlp_ratio: float
    :param qkv_bias: Add a bias to the query, key and value projections.
        Defaults to True.
    :type qkv_bias: bool
    :param dropout_rate: Dropout for the embedding and FFN layers. Defaults to
        0.0.
    :type dropout_rate: float
    :param attention_dropout_rate: Dropout on the attention weights. Defaults to
        0.0.
    :type attention_dropout_rate: float
    :param stochastic_depth_rate: Drop rate of the last block; the rate rises
        linearly from 0 across the stack. Defaults to 0.0.
    :type stochastic_depth_rate: float
    :param normalization_type: ``'layer_norm'`` or ``'rms_norm'``. Defaults to
        ``'layer_norm'``.
    :type normalization_type: str
    :param positional_embedding_type: ``'learned'`` for an absolute table,
        ``'rope'`` for 1-D rotary embeddings inside attention. Defaults to
        ``'learned'``.
    :type positional_embedding_type: str
    :param rope_theta: Rotary base frequency. Read only under ``'rope'``.
        Defaults to 10000.0.
    :type rope_theta: float
    :param rope_percentage: Fraction of each head's dimensions that are rotated.
        Read only under ``'rope'``. ``0.0`` disables the rotation, which leaves
        the model with no positional information, since ``'rope'`` creates no
        table. Defaults to 1.0.
    :type rope_percentage: float
    :param activation: FFN activation. Defaults to ``'gelu'``.
    :type activation: str or callable
    :param kernel_initializer: Initializer for kernel weights. Defaults to
        ``TruncatedNormal(stddev=0.02)``, DINO's published
        ``trunc_normal_(std=.02)``.
    :type kernel_initializer: str or dict or keras.initializers.Initializer
    :param bias_initializer: Initializer for bias weights. Defaults to
        ``'zeros'``.
    :type bias_initializer: str or keras.initializers.Initializer
    :param kernel_regularizer: Optional regularizer for kernel weights.
    :type kernel_regularizer: keras.regularizers.Regularizer or None
    :param bias_regularizer: Optional regularizer for bias weights.
    :type bias_regularizer: keras.regularizers.Regularizer or None
    :param include_top: Append the classification head. Defaults to True.
    :type include_top: bool
    :param kwargs: Additional :class:`keras.Model` arguments.

    :raises ValueError: If ``image_size`` is not divisible by ``patch_size``, if
        ``embed_dim`` is not divisible by ``num_heads``, if
        ``positional_embedding_type`` is not one of the two names, if
        ``rope_theta`` is not positive, or if ``rope_percentage`` falls outside
        ``[0, 1]``.

    Input shape:
        4D tensor ``(batch_size, height, width, channels)``, with height and
        width matching ``image_size``.

    Output shape:
        ``(batch_size, num_classes)`` when ``include_top=True`` and
        ``num_classes > 0``, otherwise ``(batch_size, embed_dim)``.
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
        # DECISION D-504: default to DINO_KERNEL_INITIALIZER, never the bare
        # string, which is Keras' stddev=0.05 not DINO's 0.02. See decisions.md.
        kernel_initializer: Union[str, Dict[str, Any], initializers.Initializer] = DINO_KERNEL_INITIALIZER,
        bias_initializer: Union[str, initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        bias_regularizer: Optional[regularizers.Regularizer] = None,
        include_top: bool = True,
        **kwargs: Any
    ) -> None:
        # Both spellings are normalized before any subscript, since v1 and v2
        # also accept a bare int.
        image_size = (
            tuple(image_size) if isinstance(image_size, (tuple, list))
            else (image_size, image_size)
        )
        patch_size = (
            tuple(patch_size) if isinstance(patch_size, (tuple, list))
            else (patch_size, patch_size)
        )

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

        # DECISION D-001: call super().__init__(inputs=, outputs=) once, at the
        # end; a bare super().__init__(**kwargs) double-inits it. decisions.md.

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

        self.num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        self.sequence_length = self.num_patches + 1

        inputs = keras.Input(shape=(*image_size, 3), name="input_image")
        outputs = self._build_model(inputs)

        # DECISION D-082: keep `name` a setdefault, never a literal; from_config
        # passes name through **kwargs and a duplicate keyword raises.
        kwargs.setdefault("name", "DINOv3")
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

        logger.info(
            f"Created DINOv3 model with {depth} layers, {embed_dim} embedding dim for "
            f"input shape {image_size}"
        )

    def _build_model(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Assemble the full graph from input tensor to output tensor."""
        x = self._build_patch_embedding(inputs)

        x = self._build_token_processing(x)

        x = self._build_encoder(x)

        x = self._build_head(x)

        return x

    def _build_patch_embedding(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Create the patch embedding layer and apply it."""
        # DECISION D-540: give every consumer its own clone_initializer copy; a
        # shared seedless instance replays its draw. See decisions.md.
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
        """Prepend the [CLS] token and add positional information."""
        # DECISION D-001: own the CLS token via ClassTokenPrepend, not an inline
        # add_weight; this runs before super().__init__. See decisions.md.
        self.cls_token_layer = ClassTokenPrepend(name="cls_token")
        x = self.cls_token_layer(x)

        # DECISION D-015: omit the learned table under 'rope', never stack it on
        # the rotation; two position signals are redundant. See decisions.md.
        if self.positional_embedding_type == 'rope':
            self.pos_embed = None
            # The 'learned' path gets its dropout from PositionalEmbedding, so
            # this path supplies its own.
            if self.dropout_rate > 0.0:
                self.embed_dropout = layers.Dropout(
                    self.dropout_rate, name="embedding_dropout"
                )
                x = self.embed_dropout(x)
            else:
                self.embed_dropout = None
            return x

        self.embed_dropout = None
        self.pos_embed = PositionalEmbedding(
            max_seq_len=self.sequence_length,
            dim=self.embed_dim,
            dropout_rate=self.dropout_rate,
            name="positional_embedding"
        )
        return self.pos_embed(x)

    def _attention_spec(self) -> Tuple[str, Dict[str, Any]]:
        """Return the attention type and factory arguments for every block.

        Under ``positional_embedding_type='rope'`` this selects the rope-capable
        ``group_query`` attention with ``num_kv_heads == num_heads``; otherwise
        the plain ``multi_head`` attention.

        :return: ``(attention_type, attention_args)``.
        :rtype: tuple
        """
        # DECISION D-015: reach RoPE through registered group_query attention,
        # never by rotating the token stream. See decisions.md.
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
        """Create the stack of transformer encoder layers and apply it."""
        self.encoder_layers = []
        # DECISION D-004: use linear_drop_path_rates, not a hand-rolled
        # ops.linspace ramp; a keras tensor has no .item(). See decisions.md.
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
                normalization_position='pre',
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
        """Apply the final normalization and, if requested, the classifier."""
        self.norm = create_normalization_layer(
            self.normalization_type,
            name='final_norm'
        )
        x = self.norm(x)

        features = x[:, 0]

        if self.include_top:
            if self.num_classes > 0:
                # DECISION D-504: DINO_KERNEL_INITIALIZER here too, never the
                # bare string, which is Keras' stddev=0.05. See decisions.md.
                self.classifier = layers.Dense(
                    units=self.num_classes,
                    kernel_initializer=initializers.get(DINO_KERNEL_INITIALIZER),
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    name='classifier'
                )
                outputs = self.classifier(features)
            else:
                outputs = features
        else:
            outputs = features

        return outputs

    def get_last_selfattention(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Raise ``NotImplementedError``; attention maps are not reachable here.

        DINO's attention-map visualization needs the last block's attention
        probabilities. ``TransformerLayer.call`` returns only the block output
        and forwards no flag that would surface them. Under ``'learned'`` the
        ``multi_head`` path accepts no such flag at all and caches nothing;
        under ``'rope'`` ``GroupedQueryAttention`` accepts
        ``return_attention_weights`` and returns a correct
        ``(batch, heads, seq, seq)`` map, but the block never passes the flag
        through. Surfacing either means changing shared layers used across the
        repository.

        :param inputs: A batch of images. Unused.
        :type inputs: keras.KerasTensor
        :raises NotImplementedError: Always.
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
        """Create a model from a named entry of ``MODEL_VARIANTS``.

        :param variant: One of ``"tiny"``, ``"small"``, ``"base"``, ``"large"``,
            ``"giant"``.
        :type variant: str
        :param image_size: Input image size; an int, or ``(height, width)``.
        :type image_size: int or tuple
        :param num_classes: Number of classifier outputs.
        :type num_classes: int
        :param include_top: Append the classification head.
        :type include_top: bool
        :param kwargs: Additional constructor arguments, which override the
            variant's own entries.
        :return: A configured model.
        :rtype: DINOv3

        :raises ValueError: If ``variant`` is not a known name.
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
        """Return every constructor argument.

        :return: Serializable configuration dictionary.
        :rtype: dict
        """
        # DECISION D-082: call super().get_config() first; without it name and
        # trainable reload at defaults, so a frozen model comes back unfrozen.
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
        """Rebuild a model from a configuration dictionary.

        :param config: Dictionary produced by :meth:`get_config`.
        :type config: dict
        :return: A configured model.
        :rtype: DINOv3
        """
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
    """Create a :class:`DINOv3` model from a variant name.

    ``create_dino_v1``, ``create_dino_v2`` and ``create_dino_v3`` share the
    signature ``(variant, *, image_size, patch_size, num_classes, include_top,
    **kwargs)``. None of them accepts ``input_shape``; the input shape comes
    from ``image_size``.

    ``patch_size=None`` defers to the variant's own entry, and an explicit
    non-``None`` value always wins. That matters for ``giant``, which carries
    ``(14, 14)`` while every other variant carries ``(16, 16)``, so
    ``create_dino_v3('giant')`` gives a /14 model and
    ``create_dino_v3('giant', patch_size=16)`` gives a /16 one. ``giant`` also
    carries ``stochastic_depth_rate=0.4``, overridable through ``**kwargs``.

    :param variant: One of ``"tiny"``, ``"small"``, ``"base"``, ``"large"``,
        ``"giant"``.
    :type variant: str
    :param image_size: Input image size; an int, or ``(height, width)``.
    :type image_size: int or tuple
    :param patch_size: Patch size; an int, or ``(height, width)``. ``None``
        defers to the variant, which is ``(14, 14)`` for ``giant`` and
        ``(16, 16)`` otherwise.
    :type patch_size: int or tuple or None
    :param num_classes: Number of classifier outputs.
    :type num_classes: int
    :param include_top: Append the classification head.
    :type include_top: bool
    :param positional_embedding_type: ``'learned'`` for an absolute table, or
        ``'rope'`` for 1-D rotary embeddings inside a ``group_query`` attention.
        A checkpoint does not transfer between the two.
    :type positional_embedding_type: str
    :param rope_theta: Rotary base frequency. Read only under ``'rope'``.
    :type rope_theta: float
    :param rope_percentage: Fraction of each head's dimensions that are rotated.
        Read only under ``'rope'``. ``0.0`` is accepted and leaves the model with
        no positional information.
    :type rope_percentage: float
    :param pretrained: Must be False; no pretrained weights ship with this
        repository.
    :type pretrained: bool
    :param kwargs: Additional constructor arguments, for example
        ``stochastic_depth_rate`` or ``normalization_type``.
    :return: A configured model.
    :rtype: DINOv3

    :raises TypeError: If ``input_shape`` is passed; use ``image_size``.
    :raises NotImplementedError: If ``pretrained=True``.
    """
    reject_input_shape(kwargs, "create_dino_v3")

    # DECISION D-069: raise on pretrained=True, do not warn and continue; no
    # checkpoints ship with this repository. See decisions.md.
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

    # DECISION D-017: patch_size=None defers to the variant; a concrete default
    # would silently turn create_dino_v3('giant') from /14 into /16.
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

# ---------------------------------------------------------------------
