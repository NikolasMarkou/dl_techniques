"""
The mask decoder's bidirectional core, built by :class:`TwoWayTransformer`:
a stack of :class:`TwoWayAttentionBlock` layers that update the query
tokens and the image features in the same pass.

Each block runs four operations in order: query self-attention,
token-to-image cross-attention, an FFN on the queries, then image-to-token
cross-attention, each a residual plus normalization. The positional
encoding is re-added at every attention rather than once at the input, so
geometry survives the residual updates across ``depth`` blocks.
``attention_downsample_rate`` (default 2) runs the three cross-attentions at
``embedding_dim // rate`` while self-attention stays at full width, matching
reference SAM; changing it changes the weight shapes, so the two settings
are not checkpoint compatible.

References:
    - Kirillov et al., 2023. Segment Anything. (https://arxiv.org/abs/2304.02643)
    - Vaswani et al., 2017. Attention Is All You Need. (https://arxiv.org/abs/1706.03762)
"""

import keras
from keras import layers, ops
from typing import Optional, Tuple, Any, Dict, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.ffn import create_ffn_layer
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.utils.activation_serialization import (
    serialize_activation,
    deserialize_activation,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# DECISION plan-2026-08-23T091307-9a110062/D-601: single home for the shipped
# 0.0 attention-dropout rate -- TwoWayTransformer(depth=2) measured 7 Dropout
# layers at this rate, so it was inert; keep it here, not restated inline. See decisions.md.
DEFAULT_ATTENTION_DROPOUT_RATE: float = 0.0

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.sam1.transformer")
class TwoWayAttentionBlock(keras.layers.Layer):
    """
    One block of bidirectional attention between sparse query tokens and
    dense image features, refining both.

    Architecture:

    .. code-block:: text

        queries [B, Nq, D]         keys [B, Nk, D]
              │                          │
              ▼                          │
        self_attn(q [+pe]) ── Add ── Norm1
              │                          │
              ▼                          ▼
        cross_attn_token_to_image(q+pe, k+pe) ── Add ── Norm2
              │
              ▼
        ffn(q) ── Add ── Norm3
              │                          │
              ▼                          ▼
        cross_attn_image_to_token(k+pe, q+pe) ── Add ── Norm4
              │                          │
              ▼                          ▼
        queries_out [B, Nq, D]     keys_out [B, Nk, D]

    :param embedding_dim: Embedding dimension for queries and keys. Must be
        divisible by ``num_heads``.
    :type embedding_dim: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param mlp_dim: Hidden dimension of the FFN. Defaults to 2048.
    :type mlp_dim: int
    :param skip_first_layer_pe: If True, skip adding positional encoding to
        the first self-attention layer, used in the transformer's first
        block. Defaults to False.
    :type skip_first_layer_pe: bool
    :param normalization_type: Normalization variant. Defaults to
        ``'layer_norm'``.
    :type normalization_type: str
    :param activation: FFN activation. Defaults to ``'relu'``.
    :type activation: str
    :param attention_dropout_rate: Dropout rate for attention layers.
        Defaults to 0.0.
    :type attention_dropout_rate: float
    :param attention_downsample_rate: Factor by which the two
        cross-attentions' internal dimension is reduced relative to
        ``embedding_dim``, while ``self_attn`` stays full width. Requires
        ``embedding_dim % (num_heads * attention_downsample_rate) == 0``.
        Defaults to 2.
    :type attention_downsample_rate: int
    :param kwargs: Additional arguments for the Layer base class.
    :ivar self_attn: Query self-attention.
    :ivar cross_attn_token_to_image: Token-to-image cross-attention.
    :ivar cross_attn_image_to_token: Image-to-token cross-attention.
    :ivar ffn: Feed-forward network for query processing.

    Input shape (in call):
        - queries: Shape (batch_size, num_queries, embedding_dim)
        - keys: Shape (batch_size, num_keys, embedding_dim)
        - query_pe: Shape (batch_size, num_queries, embedding_dim)
        - key_pe: Shape (batch_size, num_keys, embedding_dim)

    Output shape:
        Tuple of two tensors:
        - queries: Shape (batch_size, num_queries, embedding_dim)
        - keys: Shape (batch_size, num_keys, embedding_dim)

    Example:
        ```python
        block = TwoWayAttentionBlock(
            embedding_dim=256,
            num_heads=8,
            mlp_dim=2048,
            skip_first_layer_pe=False
        )

        queries = keras.random.normal(shape=(2, 10, 256))
        keys = keras.random.normal(shape=(2, 4096, 256))
        query_pe = keras.random.normal(shape=(2, 10, 256))
        key_pe = keras.random.normal(shape=(2, 4096, 256))

        queries_out, keys_out = block(queries, keys, query_pe, key_pe)
        ```
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        skip_first_layer_pe: bool = False,
        normalization_type: Literal['layer_norm', 'rms_norm', 'batch_norm'] = 'layer_norm',
        activation: str = 'relu',
        attention_dropout_rate: float = DEFAULT_ATTENTION_DROPOUT_RATE,
        attention_downsample_rate: int = 2,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        if mlp_dim <= 0:
            raise ValueError(f"mlp_dim must be positive, got {mlp_dim}")
        if not 0.0 <= attention_dropout_rate < 1.0:
            raise ValueError(f"attention_dropout_rate must be in [0, 1), got {attention_dropout_rate}")
        if attention_downsample_rate <= 0:
            raise ValueError(
                f"attention_downsample_rate must be positive, got "
                f"{attention_downsample_rate}"
            )
        if embedding_dim % (num_heads * attention_downsample_rate) != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by "
                f"num_heads * attention_downsample_rate "
                f"({num_heads} * {attention_downsample_rate} = "
                f"{num_heads * attention_downsample_rate}); otherwise the "
                f"cross-attention key_dim would be silently floored"
            )

        # Store all configuration parameters
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.skip_first_layer_pe = skip_first_layer_pe
        self.normalization_type = normalization_type
        self.activation = deserialize_activation(activation)
        self.attention_dropout_rate = attention_dropout_rate
        self.attention_downsample_rate = attention_downsample_rate

        # DECISION plan-2026-08-03T191222-1d751f81/D-009: two key dims, not
        # one -- self_attn stays full width while the cross-attentions downsample.
        # A single shared key_dim produces the same weight count but breaks official-checkpoint compatibility. See decisions.md.
        self.key_dim = embedding_dim // num_heads
        self.cross_attn_key_dim = embedding_dim // (
            num_heads * attention_downsample_rate
        )

        # CREATE all sub-layers in __init__

        # 1. Self-attention on queries
        self.self_attn = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=self.key_dim,
            dropout=attention_dropout_rate,
            name="self_attn"
        )
        self.norm1 = create_normalization_layer(
            normalization_type,
            epsilon=1e-5,
            name="norm1"
        )

        # 2. Cross-attention: tokens attending to image
        self.cross_attn_token_to_image = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=self.cross_attn_key_dim,
            dropout=attention_dropout_rate,
            name="cross_attn_token_to_image"
        )
        self.norm2 = create_normalization_layer(
            normalization_type,
            epsilon=1e-5,
            name="norm2"
        )

        # 3. Feed-forward network on queries
        self.ffn = create_ffn_layer(
            'mlp',
            hidden_dim=mlp_dim,
            output_dim=embedding_dim,
            activation=activation,
            name="ffn"
        )
        self.norm3 = create_normalization_layer(
            normalization_type,
            epsilon=1e-5,
            name="norm3"
        )

        # 4. Cross-attention: image attending to tokens
        self.cross_attn_image_to_token = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=self.cross_attn_key_dim,
            dropout=attention_dropout_rate,
            name="cross_attn_image_to_token"
        )
        self.norm4 = create_normalization_layer(
            normalization_type,
            epsilon=1e-5,
            name="norm4"
        )

    def build(self, input_shape: Optional[Tuple[Optional[int], ...]] = None) -> None:
        """
        Build every sub-layer explicitly so weights exist before
        deserialization restores them.

        :param input_shape: Not used by this layer.
        :type input_shape: Optional[Tuple[Optional[int], ...]]
        """
        self.self_attn.build(
            query_shape=(None, None, self.embedding_dim),
            value_shape=(None, None, self.embedding_dim),
            key_shape=(None, None, self.embedding_dim)
        )
        self.norm1.build((None, None, self.embedding_dim))

        # Cross-attention: tokens to image
        self.cross_attn_token_to_image.build(
            query_shape=(None, None, self.embedding_dim),
            value_shape=(None, None, self.embedding_dim),
            key_shape=(None, None, self.embedding_dim)
        )
        self.norm2.build((None, None, self.embedding_dim))

        # FFN
        self.ffn.build((None, None, self.embedding_dim))
        self.norm3.build((None, None, self.embedding_dim))

        # Cross-attention: image to tokens
        self.cross_attn_image_to_token.build(
            query_shape=(None, None, self.embedding_dim),
            value_shape=(None, None, self.embedding_dim),
            key_shape=(None, None, self.embedding_dim)
        )
        self.norm4.build((None, None, self.embedding_dim))

        super().build(input_shape)

    def call(
        self,
        queries: keras.KerasTensor,
        keys: keras.KerasTensor,
        query_pe: keras.KerasTensor,
        key_pe: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Run the four-stage bidirectional attention block.

        :param queries: Query tokens, e.g. prompt embeddings, shape
            ``(batch_size, num_queries, embedding_dim)``.
        :type queries: keras.KerasTensor
        :param keys: Key/value tokens, e.g. image features, shape
            ``(batch_size, num_keys, embedding_dim)``.
        :type keys: keras.KerasTensor
        :param query_pe: Positional encoding for queries, same shape as
            ``queries``.
        :type query_pe: keras.KerasTensor
        :param key_pe: Positional encoding for keys, same shape as ``keys``.
        :type key_pe: keras.KerasTensor
        :param training: Whether the layer runs in training mode.
        :type training: Optional[bool]
        :return: ``(updated_queries, updated_keys)``, same shapes as the
            inputs.
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor]
        """
        if self.skip_first_layer_pe:
            attn_out = self.self_attn(
                query=queries,
                value=queries,
                key=queries,
                training=training
            )
        else:
            # Subsequent blocks: add PE for self-attention
            q = queries + query_pe
            attn_out = self.self_attn(
                query=q,
                value=queries,
                key=q,
                training=training
            )
        queries = queries + attn_out
        queries = self.norm1(queries, training=training)

        # 2. Cross-attention block: tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(
            query=q,
            value=keys,
            key=k,
            training=training
        )
        queries = queries + attn_out
        queries = self.norm2(queries, training=training)

        # 3. MLP/FFN block on queries
        ffn_out = self.ffn(queries, training=training)
        queries = queries + ffn_out
        queries = self.norm3(queries, training=training)

        # 4. Cross-attention block: image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(
            query=k,
            value=queries,
            key=q,
            training=training
        )
        keys = keys + attn_out
        keys = self.norm4(keys, training=training)

        return queries, keys

    def compute_output_shape(
        self,
        queries_shape: Tuple[Optional[int], ...],
        keys_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
        """
        Compute output shapes. The argument names must be
        ``<call argument>_shape`` for every argument -- Keras 3's rule for a
        multi-argument ``compute_output_shape`` -- resolved against
        ``call(queries, keys, query_pe, key_pe)``.

        :param queries_shape: Shape of the ``queries`` argument of ``call``.
        :type queries_shape: Tuple[Optional[int], ...]
        :param keys_shape: Shape of the ``keys`` argument of ``call``.
        :type keys_shape: Tuple[Optional[int], ...]
        :return: ``(queries_output_shape, keys_output_shape)``, identical to
            the inputs.
        :rtype: Tuple[tuple, tuple]
        """
        return queries_shape, keys_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return the layer's configuration.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
            "skip_first_layer_pe": self.skip_first_layer_pe,
            "normalization_type": self.normalization_type,
            "activation": serialize_activation(self.activation),
            "attention_dropout_rate": self.attention_dropout_rate,
            "attention_downsample_rate": self.attention_downsample_rate,
        })
        return config


@register_dl_technique("dl_techniques.models.sam1.transformer")
class TwoWayTransformer(layers.Layer):
    """
    A stack of :class:`TwoWayAttentionBlock` layers plus a final
    query-to-image cross-attention, for joint refinement of queries and
    image features.

    Architecture:

    .. code-block:: text

        image_embedding [B, H, W, D]     point_embedding [B, N, D]
              │                                │
              ▼                                │
        flatten to [B, H*W, D]                 │
              │                                │
              └──────► TwoWayAttentionBlock x depth ◄──────┘
                              │
                              ▼
                final_attn_token_to_image ── Add ── norm_final_attn
                              │
                              ▼
              queries [B, N, D], keys [B, H*W, D]

    :param depth: Number of :class:`TwoWayAttentionBlock` layers. Must be
        positive.
    :type depth: int
    :param embedding_dim: Embedding dimension throughout the transformer.
        Must be divisible by ``num_heads``.
    :type embedding_dim: int
    :param num_heads: Number of attention heads per layer.
    :type num_heads: int
    :param mlp_dim: Hidden dimension of the FFNs. Defaults to 2048.
    :type mlp_dim: int
    :param normalization_type: Normalization variant. Defaults to
        ``'layer_norm'``.
    :type normalization_type: str
    :param activation: FFN activation. Defaults to ``'relu'``.
    :type activation: str
    :param attention_dropout_rate: Dropout rate for attention layers.
        Defaults to 0.0.
    :type attention_dropout_rate: float
    :param attention_downsample_rate: Forwarded to every block and to
        ``final_attn_token_to_image``. Requires
        ``embedding_dim % (num_heads * attention_downsample_rate) == 0``.
        Defaults to 2.
    :type attention_downsample_rate: int
    :param kwargs: Additional arguments for the Layer base class.
    :ivar layers_list: List of :class:`TwoWayAttentionBlock` instances.
    :ivar final_attn_token_to_image: Final query-to-image attention.

    Input shape (in call):
        - image_embedding: Shape (batch_size, H, W, embedding_dim)
        - image_pe: Shape (batch_size, H, W, embedding_dim)
        - point_embedding: Shape (batch_size, num_points, embedding_dim)

    Output shape:
        Tuple of two tensors:
        - queries: Shape (batch_size, num_points, embedding_dim)
        - keys: Shape (batch_size, H*W, embedding_dim)

    Example:
        ```python
        transformer = TwoWayTransformer(
            depth=2,
            embedding_dim=256,
            num_heads=8,
            mlp_dim=2048
        )

        image_emb = keras.random.normal(shape=(2, 64, 64, 256))
        image_pe = keras.random.normal(shape=(2, 64, 64, 256))
        point_emb = keras.random.normal(shape=(2, 5, 256))

        queries, keys = transformer(image_emb, image_pe, point_emb)
        print(f"Queries: {queries.shape}")  # (2, 5, 256)
        print(f"Keys: {keys.shape}")        # (2, 4096, 256)
        ```

    Note:
        The first block runs with ``skip_first_layer_pe=True``, matching
        reference SAM.
    """

    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        normalization_type: Literal['layer_norm', 'rms_norm', 'batch_norm'] = 'layer_norm',
        activation: str = 'relu',
        attention_dropout_rate: float = DEFAULT_ATTENTION_DROPOUT_RATE,
        attention_downsample_rate: int = 2,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        if mlp_dim <= 0:
            raise ValueError(f"mlp_dim must be positive, got {mlp_dim}")
        if attention_downsample_rate <= 0:
            raise ValueError(
                f"attention_downsample_rate must be positive, got "
                f"{attention_downsample_rate}"
            )
        if embedding_dim % (num_heads * attention_downsample_rate) != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by "
                f"num_heads * attention_downsample_rate "
                f"({num_heads} * {attention_downsample_rate} = "
                f"{num_heads * attention_downsample_rate}); otherwise the "
                f"cross-attention key_dim would be silently floored"
            )

        # Store all configuration parameters
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.normalization_type = normalization_type
        self.activation = deserialize_activation(activation)
        self.attention_dropout_rate = attention_dropout_rate
        self.attention_downsample_rate = attention_downsample_rate

        # DECISION plan-2026-08-03T191222-1d751f81/D-009: final_attn_token_to_image
        # is a cross attention, so it takes the downsampled key_dim, not
        # self.key_dim -- self.key_dim is kept only as public attribute surface. See decisions.md.
        self.key_dim = embedding_dim // num_heads
        self.cross_attn_key_dim = embedding_dim // (
            num_heads * attention_downsample_rate
        )

        # CREATE all sub-layers in __init__

        # Stack of two-way attention blocks
        self.layers_list = []
        for i in range(depth):
            block = TwoWayAttentionBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                skip_first_layer_pe=(i == 0),  # First block skips PE in self-attention
                normalization_type=normalization_type,
                activation=activation,
                attention_dropout_rate=attention_dropout_rate,
                attention_downsample_rate=attention_downsample_rate,
                name=f"block_{i}"
            )
            self.layers_list.append(block)

        # Final attention: queries attend to refined image features
        self.final_attn_token_to_image = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=self.cross_attn_key_dim,
            dropout=attention_dropout_rate,
            name="final_attn_token_to_image"
        )
        self.norm_final_attn = create_normalization_layer(
            normalization_type,
            epsilon=1e-5,
            name="norm_final_attn"
        )

    def build(self, input_shape: Optional[Tuple[Optional[int], ...]] = None) -> None:
        """
        Build every sub-layer explicitly so weights exist before
        deserialization restores them.

        :param input_shape: Not used by this layer.
        :type input_shape: Optional[Tuple[Optional[int], ...]]
        """
        for block in self.layers_list:
            block.build(None)

        self.final_attn_token_to_image.build(
            query_shape=(None, None, self.embedding_dim),
            value_shape=(None, None, self.embedding_dim),
            key_shape=(None, None, self.embedding_dim)
        )
        self.norm_final_attn.build((None, None, self.embedding_dim))

        super().build(input_shape)

    def call(
            self,
            image_embedding: keras.KerasTensor,
            image_pe: keras.KerasTensor,
            point_embedding: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Flatten the image features, run the block stack, then a final
        query-to-image cross-attention.

        :param image_embedding: Image features from the encoder, shape
            ``(batch_size, H, W, embedding_dim)``.
        :type image_embedding: keras.KerasTensor
        :param image_pe: Positional encoding for the image, same shape as
            ``image_embedding``.
        :type image_pe: keras.KerasTensor
        :param point_embedding: Query tokens (prompts + output tokens),
            shape ``(batch_size, num_queries, embedding_dim)``.
        :type point_embedding: keras.KerasTensor
        :param training: Whether the layer runs in training mode.
        :type training: Optional[bool]
        :return: ``(refined_queries, refined_image_features)`` of shapes
            ``(batch_size, num_queries, embedding_dim)`` and
            ``(batch_size, H*W, embedding_dim)``.
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor]
        """
        B, H, W, C = ops.shape(image_embedding)
        image_embedding_flat = ops.reshape(image_embedding, (B, H * W, C))

        image_pe = ops.broadcast_to(image_pe, (B, H, W, C))
        image_pe_flat = ops.reshape(image_pe, (B, H * W, C))

        queries = point_embedding
        keys = image_embedding_flat

        # The original point embedding is used as the query PE in every
        # block, not the running `queries`.
        for layer in self.layers_list:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe_flat,
                training=training
            )

        q = queries + point_embedding
        k = keys + image_pe_flat
        attn_out = self.final_attn_token_to_image(
            query=q,
            value=keys,
            key=k,
            training=training
        )
        queries = queries + attn_out
        queries = self.norm_final_attn(queries, training=training)

        return queries, keys

    def compute_output_shape(
        self,
        image_embedding_shape: Tuple[Optional[int], ...],
        point_embedding_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
        """
        Compute output shapes. The argument names must be
        ``<call argument>_shape`` for every argument -- Keras 3's rule for a
        multi-argument ``compute_output_shape`` -- resolved against
        ``call(image_embedding, image_pe, point_embedding)``.

        :param image_embedding_shape: Shape of ``image_embedding``,
            ``(B, H, W, C)``.
        :type image_embedding_shape: Tuple[Optional[int], ...]
        :param point_embedding_shape: Shape of ``point_embedding``,
            ``(B, N, C)``.
        :type point_embedding_shape: Tuple[Optional[int], ...]
        :return: ``(query_shape, key_shape)`` -- query is
            ``point_embedding_shape``, key is the flattened image shape
            ``(B, H*W, C)``.
        :rtype: Tuple[tuple, tuple]
        """
        batch_size = point_embedding_shape[0]
        num_queries = point_embedding_shape[1]
        embedding_dim = point_embedding_shape[2]

        if image_embedding_shape[1] is not None and image_embedding_shape[2] is not None:
            num_keys = image_embedding_shape[1] * image_embedding_shape[2]
        else:
            num_keys = None

        query_shape = (batch_size, num_queries, embedding_dim)
        key_shape = (batch_size, num_keys, embedding_dim)

        return query_shape, key_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return the layer's configuration.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "depth": self.depth,
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
            "normalization_type": self.normalization_type,
            "activation": serialize_activation(self.activation),
            "attention_dropout_rate": self.attention_dropout_rate,
            "attention_downsample_rate": self.attention_downsample_rate,
        })
        return config

# ---------------------------------------------------------------------
