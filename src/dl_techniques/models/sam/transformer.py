"""
SAM Two-Way Transformer Implementation
=================================================

This file provides a Keras 3 implementation of the Two-Way Transformer used
in the Segment Anything Model's mask decoder. This is a specialized decoder
architecture that bidirectionally updates both token and image embeddings.

**Intent**: To create a faithful and serializable Keras implementation of the
SAM transformer architecture, following modern Keras best practices for building
complex, composite layers with factory pattern integration.

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
Queries_in (tokens), Keys_in (image features)
    │
    v
┌─────────────────────────────────────────┐
│ 1. Self-Attention on Queries            │
│    Q, K, V = Queries + PE               │
│    Queries' = Queries + Attention(Q,K,V)│
│    Queries' = Norm(Queries')            │
└─────────────────────────────────────────┘
    │
    v
┌───────────────────────────────────────────┐
│ 2. Cross-Attention (Token to Image)       │
│    Q = Queries' + PE                      │
│    K, V = Keys + PE, Keys                 │
│    Queries'' = Queries' + Attention(Q,K,V)│
│    Queries'' = Norm(Queries'')            │
└───────────────────────────────────────────┘
    │
    v
┌───────────────────────────────────────────┐
│ 3. MLP/FFN on Queries                     │
│    Queries''' = Queries'' + FFN(Queries'')│
│    Queries''' = Norm(Queries''')          │
└───────────────────────────────────────────┘
    │
    v
┌─────────────────────────────────────────┐
│ 4. Cross-Attention (Image to Token)     │
│    Q = Keys + PE                        │
│    K, V = Queries''' + PE, Queries'''   │
│    Keys' = Keys + Attention(Q,K,V)      │
│    Keys' = Norm(Keys')                  │
└─────────────────────────────────────────┘
    │
    v
Queries_out, Keys_out
```

**Usage Example**:
```python
import keras

# Create two-way transformer
transformer = TwoWayTransformer(
    depth=2,
    embedding_dim=256,
    num_heads=8,
    mlp_dim=2048,
    normalization_type='layer_norm',
    activation='relu'
)

# Prepare inputs
image_embedding = keras.random.normal(shape=(1, 64, 64, 256))
image_pe = keras.random.normal(shape=(1, 64, 64, 256))
point_embedding = keras.random.normal(shape=(1, 5, 256))

# Run transformer
queries_out, keys_out = transformer(image_embedding, image_pe, point_embedding)

print(f"Queries output shape: {queries_out.shape}")  # (1, 5, 256)
print(f"Keys output shape: {keys_out.shape}")        # (1, 64*64, 256)
```

**References**:
- Kirillov, A., et al. (2023). Segment Anything. *arXiv*.
- Vaswani, A., et al. (2017). Attention is All You Need. *NeurIPS*.
"""

import keras
from keras import layers, ops
from typing import Optional, Tuple, Any, Dict, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.ffn import create_ffn_layer
from dl_techniques.layers.norms import create_normalization_layer

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class TwoWayAttentionBlock(keras.layers.Layer):
    """
    A transformer block with four layers for bidirectional attention.

    This block implements the core processing unit of the Two-Way Transformer,
    performing self-attention on queries, bidirectional cross-attention between
    queries and keys (image features), and feed-forward processing.

    **Intent**: To enable bidirectional information flow between sparse tokens
    (prompts) and dense image embeddings, allowing both to be refined based on
    each other.

    **Architecture**:
    The block consists of four sequential operations:
    1. Self-attention on sparse queries (tokens)
    2. Cross-attention from queries to image features
    3. Feed-forward network on queries
    4. Cross-attention from image features to queries

    Each operation includes a residual connection and normalization.

    Args:
        embedding_dim: Integer, the embedding dimension for queries and keys.
            Must be divisible by num_heads.
        num_heads: Integer, number of attention heads. Must divide embedding_dim
            evenly.
        mlp_dim: Integer, hidden dimension of the feed-forward network.
            Defaults to 2048.
        skip_first_layer_pe: Boolean, if True, skips adding positional encoding
            to the first self-attention layer. Used in the first block of the
            transformer. Defaults to False.
        normalization_type: String, type of normalization to use. Supports
            'layer_norm', 'rms_norm', 'batch_norm'. Defaults to 'layer_norm'.
        activation: String, activation function for FFN. Defaults to 'relu'.
        attention_dropout: Float, dropout rate for attention layers.
            Defaults to 0.0.
        **kwargs: Additional arguments for the Layer base class.

    Input shape (in call):
        - queries: Shape (batch_size, num_queries, embedding_dim)
        - keys: Shape (batch_size, num_keys, embedding_dim)
        - query_pe: Shape (batch_size, num_queries, embedding_dim)
        - key_pe: Shape (batch_size, num_keys, embedding_dim)

    Output shape:
        Tuple of two tensors:
        - queries: Shape (batch_size, num_queries, embedding_dim)
        - keys: Shape (batch_size, num_keys, embedding_dim)

    Attributes:
        self_attn: MultiHeadAttention layer for query self-attention.
        cross_attn_token_to_image: MultiHeadAttention for token->image attention.
        cross_attn_image_to_token: MultiHeadAttention for image->token attention.
        ffn: Feed-forward network for query processing.
        norm1, norm2, norm3, norm4: Normalization layers for each sub-layer.

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
        attention_dropout: float = 0.0,
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
        if not 0.0 <= attention_dropout < 1.0:
            raise ValueError(f"attention_dropout must be in [0, 1), got {attention_dropout}")

        # Store all configuration parameters
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.skip_first_layer_pe = skip_first_layer_pe
        self.normalization_type = normalization_type
        self.activation = activation
        self.attention_dropout = attention_dropout

        # Calculate key dimension for attention
        self.key_dim = embedding_dim // num_heads

        # CREATE all sub-layers in __init__

        # 1. Self-attention on queries
        self.self_attn = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=self.key_dim,
            dropout=attention_dropout,
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
            key_dim=self.key_dim,
            dropout=attention_dropout,
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
            key_dim=self.key_dim,
            dropout=attention_dropout,
            name="cross_attn_image_to_token"
        )
        self.norm4 = create_normalization_layer(
            normalization_type,
            epsilon=1e-5,
            name="norm4"
        )

    def build(self, input_shape: Optional[Tuple[Optional[int], ...]] = None) -> None:
        """
        Builds all sub-layers.

        Following the "Create vs. Build" principle, we explicitly build all
        sub-layers to ensure their weights are created before the model attempts
        to load any saved weights during deserialization.

        Args:
            input_shape: Optional shape tuple (not used for this layer).
        """
        # Build attention layers
        # Self-attention: queries attend to queries
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
        Forward pass through the two-way attention block.

        Args:
            queries: Query tokens (e.g., prompt embeddings), shape
                (batch_size, num_queries, embedding_dim).
            keys: Key/value tokens (e.g., image features), shape
                (batch_size, num_keys, embedding_dim).
            query_pe: Positional encoding for queries, shape
                (batch_size, num_queries, embedding_dim).
            key_pe: Positional encoding for keys, shape
                (batch_size, num_keys, embedding_dim).
            training: Optional boolean for training mode.

        Returns:
            Tuple of (updated_queries, updated_keys):
            - updated_queries: Shape (batch_size, num_queries, embedding_dim)
            - updated_keys: Shape (batch_size, num_keys, embedding_dim)
        """

        # 1. Self-attention block on queries
        if self.skip_first_layer_pe:
            # First block: don't add PE to self-attention
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
        query_shape: Tuple[Optional[int], ...],
        key_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
        """
        Compute output shapes.

        Args:
            query_shape: Shape of queries.
            key_shape: Shape of keys.

        Returns:
            Tuple of (query_output_shape, key_output_shape), which are
            identical to the input shapes.
        """
        return query_shape, key_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the configuration of the layer for serialization.

        Returns:
            Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
            "skip_first_layer_pe": self.skip_first_layer_pe,
            "normalization_type": self.normalization_type,
            "activation": self.activation,
            "attention_dropout": self.attention_dropout,
        })
        return config


@keras.saving.register_keras_serializable()
class TwoWayTransformer(layers.Layer):
    """
    A two-way transformer decoder for joint refinement of queries and image features.

    This transformer implements bidirectional attention between sparse queries
    (e.g., prompt tokens) and dense image features. It consists of multiple
    TwoWayAttentionBlock layers followed by a final attention layer that allows
    queries to attend to the refined image features one last time.

    **Intent**: To serve as the core processing module in SAM's mask decoder,
    enabling prompts to gather information from the image while simultaneously
    allowing the image representation to be refined based on the prompts.

    **Architecture**:
    ```
    Input: Image Embeddings + Positional Encoding, Point/Prompt Embeddings

    For each of 'depth' layers:
        ┌─────────────────────────────────┐
        │  TwoWayAttentionBlock           │
        │  - Self-attention on queries    │
        │  - Cross-attention (Q→Img)      │
        │  - FFN on queries               │
        │  - Cross-attention (Img→Q)      │
        └─────────────────────────────────┘
                    ↓
    ┌─────────────────────────────────────┐
    │  Final Cross-Attention              │
    │  Queries attend to refined image    │
    └─────────────────────────────────────┘
                    ↓
    Output: Refined Queries, Refined Image Features
    ```

    Args:
        depth: Integer, number of TwoWayAttentionBlock layers. Must be positive.
        embedding_dim: Integer, the embedding dimension for all inputs and
            throughout the transformer. Must be divisible by num_heads.
        num_heads: Integer, number of attention heads in each attention layer.
        mlp_dim: Integer, hidden dimension of the feed-forward networks.
            Defaults to 2048.
        normalization_type: String, type of normalization to use. Supports
            'layer_norm', 'rms_norm', 'batch_norm'. Defaults to 'layer_norm'.
        activation: String, activation function for FFN. Defaults to 'relu'.
        attention_dropout: Float, dropout rate for attention layers.
            Defaults to 0.0.
        **kwargs: Additional arguments for the Layer base class.

    Input shape (in call):
        - image_embedding: Shape (batch_size, H, W, embedding_dim)
        - image_pe: Shape (batch_size, H, W, embedding_dim)
        - point_embedding: Shape (batch_size, num_points, embedding_dim)

    Output shape:
        Tuple of two tensors:
        - queries: Shape (batch_size, num_points, embedding_dim)
        - keys: Shape (batch_size, H*W, embedding_dim)

    Attributes:
        layers: List of TwoWayAttentionBlock instances.
        final_attn_token_to_image: Final attention layer for queries to image.
        norm_final_attn: Normalization after final attention.

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
        The first TwoWayAttentionBlock has skip_first_layer_pe=True, meaning
        it doesn't add positional encoding to its self-attention. This is a
        design choice from the original SAM architecture.
    """

    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        normalization_type: Literal['layer_norm', 'rms_norm', 'batch_norm'] = 'layer_norm',
        activation: str = 'relu',
        attention_dropout: float = 0.0,
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

        # Store all configuration parameters
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.normalization_type = normalization_type
        self.activation = activation
        self.attention_dropout = attention_dropout

        # Calculate key dimension for attention
        self.key_dim = embedding_dim // num_heads

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
                attention_dropout=attention_dropout,
                name=f"block_{i}"
            )
            self.layers_list.append(block)

        # Final attention: queries attend to refined image features
        self.final_attn_token_to_image = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=self.key_dim,
            dropout=attention_dropout,
            name="final_attn_token_to_image"
        )
        self.norm_final_attn = create_normalization_layer(
            normalization_type,
            epsilon=1e-5,
            name="norm_final_attn"
        )

    def build(self, input_shape: Optional[Tuple[Optional[int], ...]] = None) -> None:
        """
        Builds all sub-layers.

        Following the "Create vs. Build" principle, we explicitly build all
        sub-layers to ensure their weights are created before the model attempts
        to load any saved weights during deserialization.

        Args:
            input_shape: Optional shape tuple (not used for this layer).
        """
        # Build all two-way attention blocks
        for block in self.layers_list:
            block.build(None)

        # Build final attention layer
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
        Forward pass through the two-way transformer.

        Args:
            image_embedding: Image features from encoder, shape
                (batch_size, H, W, embedding_dim).
            image_pe: Positional encoding for image, shape
                (batch_size, H, W, embedding_dim).
            point_embedding: Query tokens (prompts + output tokens), shape
                (batch_size, num_queries, embedding_dim).
            training: Optional boolean for training mode.

        Returns:
            Tuple of (refined_queries, refined_image_features):
            - refined_queries: Shape (batch_size, num_queries, embedding_dim)
            - refined_image_features: Shape (batch_size, H*W, embedding_dim)
        """
        # Reshape image embeddings from (B, H, W, C) to (B, H*W, C) for attention
        B, H, W, C = ops.shape(image_embedding)
        image_embedding_flat = ops.reshape(image_embedding, (B, H * W, C))

        # Broadcast image PE to match batch size and reshape
        image_pe = ops.broadcast_to(image_pe, (B, H, W, C))
        image_pe_flat = ops.reshape(image_pe, (B, H * W, C))

        # Initialize queries and keys
        queries = point_embedding
        keys = image_embedding_flat

        # Process through stack of two-way attention blocks
        for layer in self.layers_list:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,  # Use original point embedding as PE throughout
                key_pe=image_pe_flat,
                training=training
            )

        # Final attention: queries attend to refined image features one more time
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
        image_shape: Tuple[Optional[int], ...],
        point_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
        """
        Compute output shapes.

        Args:
            image_shape: Shape of image_embedding (B, H, W, C).
            point_shape: Shape of point_embedding (B, N, C).

        Returns:
            Tuple of (query_shape, key_shape):
            - query_shape: (B, N, C) - same as point_shape
            - key_shape: (B, H*W, C) - flattened image shape
        """
        batch_size = point_shape[0]
        num_queries = point_shape[1]
        embedding_dim = point_shape[2]

        # Image is flattened spatially
        if image_shape[1] is not None and image_shape[2] is not None:
            num_keys = image_shape[1] * image_shape[2]
        else:
            num_keys = None

        query_shape = (batch_size, num_queries, embedding_dim)
        key_shape = (batch_size, num_keys, embedding_dim)

        return query_shape, key_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the configuration of the layer for serialization.

        Returns:
            Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            "depth": self.depth,
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
            "normalization_type": self.normalization_type,
            "activation": self.activation,
            "attention_dropout": self.attention_dropout,
        })
        return config

# ---------------------------------------------------------------------
