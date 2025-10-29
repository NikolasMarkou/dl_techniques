"""
SAM Two-Way Transformer Implementation in Keras 3
=================================================

This file provides a Keras 3 implementation of the Two-Way Transformer used
in the Segment Anything Model's mask decoder. This is a specialized decoder
architecture that bidirectionally updates both token and image embeddings.

**Intent**: To create a faithful and serializable Keras implementation of the
SAM transformer architecture, following modern Keras best practices for building
complex, composite layers.

**Architecture**: The Two-Way Transformer consists of a series of
`TwoWayAttentionBlock` layers. Each block performs four main operations:

1.  **Query Self-Attention**: Queries (prompt and mask tokens) attend to
    themselves.
2.  **Cross-Attention (Token to Image)**: Queries attend to the image embeddings.
3.  **MLP on Queries**: A standard feed-forward network is applied to the
    updated queries.
4.  **Cross-Attention (Image to Token)**: Image embeddings attend to the
    updated queries.

This bidirectional flow allows prompts to gather information from the image and
the image representation to be refined based on the prompts.

**Data Flow (per block)**:
```
Queries_in, Keys_in (Image Embeddings)
    |
    v
Self-Attention(Queries_in) -> Queries'
    |
    v
Cross-Attention(Q=Queries', K=Keys_in, V=Keys_in) -> Queries''
    |
    v
MLP(Queries'') -> Queries'''
    |
    v
Cross-Attention(Q=Keys_in, K=Queries''', V=Queries''') -> Keys'
    |
    v
Queries_out, Keys_out
```
"""

import keras
from keras import layers, ops
from typing import Optional, Tuple, Type, List, Any, Dict, Union

@keras.saving.register_keras_serializable()
class TwoWayAttentionBlock(keras.layers.Layer):
    """
    A transformer block with four layers: self-attention of sparse inputs,
    cross-attention of sparse to dense inputs, an MLP on sparse inputs,
    and cross-attention of dense to sparse inputs.
    """

    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            mlp_dim: int = 2048,
            skip_first_layer_pe: bool = False,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.skip_first_layer_pe = skip_first_layer_pe

        self.self_attn = layers.MultiHeadAttention(num_heads, embedding_dim // num_heads, name="self_attn")
        self.norm1 = layers.LayerNormalization(epsilon=1e-5, name="norm1")
        self.cross_attn_token_to_image = layers.MultiHeadAttention(num_heads, embedding_dim // num_heads,
                                                                   name="cross_attn_t2i")
        self.norm2 = layers.LayerNormalization(epsilon=1e-5, name="norm2")
        self.ffn = create_ffn_layer('mlp', hidden_dim=mlp_dim, output_dim=embedding_dim, activation='relu', name="ffn")
        self.norm3 = layers.LayerNormalization(epsilon=1e-5, name="norm3")
        self.norm4 = layers.LayerNormalization(epsilon=1e-5, name="norm4")
        self.cross_attn_image_to_token = layers.MultiHeadAttention(num_heads, embedding_dim // num_heads,
                                                                   name="cross_attn_i2t")

    def call(self, queries: keras.KerasTensor, keys: keras.KerasTensor, query_pe: keras.KerasTensor,
             key_pe: keras.KerasTensor) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        # Self-attention block
        if self.skip_first_layer_pe:
            queries = queries + self.self_attn(query=queries, value=queries, key=queries)
        else:
            q = queries + query_pe
            queries = queries + self.self_attn(query=q, value=queries, key=q)
        queries = self.norm1(queries)

        # Cross-attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        queries = queries + self.cross_attn_token_to_image(query=q, value=keys, key=k)
        queries = self.norm2(queries)

        # MLP block
        queries = queries + self.ffn(queries)
        queries = self.norm3(queries)

        # Cross-attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        keys = keys + self.cross_attn_image_to_token(query=k, value=queries, key=q)
        keys = self.norm4(keys)

        return queries, keys

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
            "skip_first_layer_pe": self.skip_first_layer_pe,
        })
        return config


@keras.saving.register_keras_serializable()
class TwoWayTransformer(layers.Layer):
    """
    A transformer decoder that attends to an input image using queries whose
    positional embedding is supplied.
    """

    def __init__(self, depth: int, embedding_dim: int, num_heads: int, mlp_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim

        self.layers = [
            TwoWayAttentionBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                skip_first_layer_pe=(i == 0),
                name=f"block_{i}"
            )
            for i in range(depth)
        ]
        self.final_attn_token_to_image = layers.MultiHeadAttention(num_heads, embedding_dim // num_heads,
                                                                   name="final_attn")
        self.norm_final_attn = layers.LayerNormalization(epsilon=1e-5, name="norm_final_attn")

    def call(self, image_embedding: keras.KerasTensor, image_pe: keras.KerasTensor,
             point_embedding: keras.KerasTensor) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        # Reshape image embeddings for attention
        B, H, W, C = ops.shape(image_embedding)
        image_embedding = ops.reshape(image_embedding, (B, H * W, C))
        image_pe = ops.reshape(image_pe, (B, H * W, C))

        queries = point_embedding
        keys = image_embedding

        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(query=q, value=keys, key=k)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "depth": self.depth,
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
        })
        return config

