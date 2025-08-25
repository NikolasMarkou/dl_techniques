"""
Encoder-only Mask Transformer (EoMT) Layer Implementation

Based on: "Your ViT is Secretly an Image Segmentation Model" by Kerssies et al.

This module provides modern Keras 3 implementations of EoMT layers following
best practices for custom layer development, serialization, and robustness.
"""

import keras
from keras import ops, layers, initializers, regularizers, activations
from typing import Optional, Any, Tuple, Union, Dict, Callable
import numpy as np

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class EoMTLayer(keras.layers.Layer):
    """
    Encoder-only Mask Transformer layer for vision segmentation tasks.

    This layer implements the core EoMT architecture that processes both patch tokens
    and learnable query tokens together using Vision Transformer self-attention. It
    supports masked attention during training with configurable probability annealing,
    enabling the model to learn object-centric representations for segmentation.

    **Intent**: Provide a transformer layer that can jointly process image patches
    and learned queries for instance segmentation, combining the strengths of
    transformer architectures with object-centric learning approaches.

    **Architecture**:
    ```
    Input: [patches; queries] → shape [batch, seq_len, embed_dim]
           ↓
    LayerNorm₁ (optional)
           ↓
    Multi-Head Self-Attention (with optional masking)
      - QKV projection: embed_dim → 3 * embed_dim
      - Attention computation with scaled dot-product
      - Masked attention: query-to-patch interactions filtered by mask
      - Output projection: embed_dim → embed_dim
           ↓
    Residual Connection₁
           ↓
    LayerNorm₂ (optional)
           ↓
    MLP Block:
      - Dense₁: embed_dim → mlp_ratio * embed_dim (with activation)
      - Dropout₁
      - Dense₂: mlp_ratio * embed_dim → embed_dim
      - Dropout₂
           ↓
    Residual Connection₂
           ↓
    Output: [patches; queries] → shape [batch, seq_len, embed_dim]
    ```

    **Masked Attention Mechanism**:
    During training, query-to-patch attention can be masked based on ground truth
    segmentation masks. This encourages queries to attend only to relevant patches,
    promoting object-centric learning. The masking is applied probabilistically
    and can be annealed during training.

    **Mathematical Operations**:
    1. **Self-Attention**: Attention(Q, K, V) = softmax(QK^T/√d)V
    2. **Masked Attention**: For query i and patch j, if mask[i,j] = 0 then attention[i,j] = -∞
    3. **MLP**: MLP(x) = Dense₂(dropout(activation(Dense₁(x))))
    4. **Residual**: output = x + sublayer(norm(x))

    Args:
        embed_dim: Integer, embedding dimension for all tokens. Must be positive
            and divisible by num_heads. This is the model's hidden size.
        num_heads: Integer, number of attention heads. Must be positive and
            divide evenly into embed_dim. Defaults to 8.
        mlp_ratio: Float, ratio of MLP hidden dimension to embedding dimension.
            Controls the expansion in the feed-forward network. Defaults to 4.0.
        dropout: Float between 0 and 1, dropout rate applied to MLP layers and
            attention output projection. Defaults to 0.0.
        attention_dropout: Float between 0 and 1, dropout rate applied to
            attention weights. Defaults to 0.0.
        use_layer_norm: Boolean, whether to apply layer normalization before
            attention and MLP blocks. Defaults to True.
        activation: String or callable, activation function for the MLP block.
            Defaults to 'gelu'.
        use_masked_attention: Boolean, whether to enable masked attention during
            training. When True, requires mask input during forward pass.
            Defaults to False.
        mask_probability: Float between 0 and 1, probability of applying masked
            attention when use_masked_attention=True. Used for curriculum learning
            or annealing. Defaults to 1.0.
        kernel_initializer: String or initializer, weight initialization for
            dense layers. Defaults to 'glorot_uniform'.
        bias_initializer: String or initializer, bias initialization for
            dense layers. Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer for dense layer weights.
        bias_regularizer: Optional regularizer for dense layer biases.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        3D tensor with shape: `(batch_size, seq_len, embed_dim)`
        where seq_len = num_patches + num_queries

    Output shape:
        3D tensor with shape: `(batch_size, seq_len, embed_dim)`
        Same shape as input.

    Attributes:
        norm1: First LayerNormalization layer (if use_layer_norm=True).
        norm2: Second LayerNormalization layer (if use_layer_norm=True).
        qkv: Dense layer for computing queries, keys, and values.
        proj: Output projection layer after attention.
        mlp: Sequential MLP block for feed-forward processing.
        dropout_layer: Dropout layer for attention output (if dropout > 0).
        attention_dropout_layer: Dropout layer for attention weights (if attention_dropout > 0).

    Example:
        ```python
        # Basic usage for standard vision transformer
        layer = EoMTLayer(embed_dim=768, num_heads=12)
        inputs = keras.Input(shape=(197, 768))  # 196 patches + 1 CLS token
        outputs = layer(inputs)

        # With masked attention for segmentation training
        layer = EoMTLayer(
            embed_dim=768,
            num_heads=12,
            use_masked_attention=True,
            mask_probability=0.8  # 80% chance of applying mask
        )

        # During training with mask
        patches = keras.Input(shape=(196, 768))
        queries = keras.Input(shape=(100, 768))
        tokens = keras.layers.Concatenate(axis=1)([patches, queries])
        mask = keras.Input(shape=(100, 14, 14))  # Query masks for 14x14 patches

        outputs = layer(tokens, mask=mask)

        # Curriculum learning: gradually increase mask probability
        for epoch in range(num_epochs):
            layer.mask_probability = min(1.0, epoch / 10)  # Ramp up over 10 epochs
        ```

    References:
        - "Your ViT is Secretly a Segmentation Model": https://arxiv.org/abs/2312.02113
        - "Attention Is All You Need": https://arxiv.org/abs/1706.03762
        - "An Image is Worth 16x16 Words": https://arxiv.org/abs/2010.11929

    Note:
        When using masked attention, the mask tensor shape should be
        [batch_size, num_queries, H, W] where H*W equals num_patches.
        The layer automatically determines the split between patches and queries
        based on the mask dimensions.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        use_layer_norm: bool = True,
        activation: Union[str, Callable] = "gelu",
        use_masked_attention: bool = False,
        mask_probability: float = 1.0,
        kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
        bias_initializer: Union[str, initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        bias_regularizer: Optional[regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        if not (0.0 <= dropout <= 1.0):
            raise ValueError(f"dropout must be between 0 and 1, got {dropout}")
        if not (0.0 <= attention_dropout <= 1.0):
            raise ValueError(f"attention_dropout must be between 0 and 1, got {attention_dropout}")
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")
        if not (0.0 <= mask_probability <= 1.0):
            raise ValueError(f"mask_probability must be between 0 and 1, got {mask_probability}")

        # Store configuration
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.use_layer_norm = use_layer_norm
        self.activation = activations.get(activation)
        self.use_masked_attention = use_masked_attention
        self.mask_probability = mask_probability
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        # Computed attributes
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # CREATE all sub-layers in __init__ (following modern Keras 3 pattern)

        # Normalization layers
        if self.use_layer_norm:
            self.norm1 = layers.LayerNormalization(
                epsilon=1e-6,
                name="norm1"
            )
            self.norm2 = layers.LayerNormalization(
                epsilon=1e-6,
                name="norm2"
            )
        else:
            self.norm1 = None
            self.norm2 = None

        # Attention layers
        self.qkv = layers.Dense(
            self.embed_dim * 3,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="qkv"
        )

        self.proj = layers.Dense(
            self.embed_dim,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="proj"
        )

        # MLP block
        mlp_hidden_dim = int(self.embed_dim * self.mlp_ratio)
        self.mlp = keras.Sequential([
            layers.Dense(
                mlp_hidden_dim,
                activation=self.activation,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name="mlp_fc1"
            ),
            layers.Dropout(self.dropout, name="mlp_dropout1"),
            layers.Dense(
                self.embed_dim,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name="mlp_fc2"
            ),
            layers.Dropout(self.dropout, name="mlp_dropout2")
        ], name="mlp")

        # Dropout layers
        if self.dropout > 0:
            self.dropout_layer = layers.Dropout(self.dropout, name="dropout")
        else:
            self.dropout_layer = None

        if self.attention_dropout > 0:
            self.attention_dropout_layer = layers.Dropout(
                self.attention_dropout, name="attention_dropout"
            )
        else:
            self.attention_dropout_layer = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers.

        CRITICAL: Explicitly build each sub-layer for robust serialization.
        """
        # Build normalization layers if they exist
        if self.norm1 is not None:
            self.norm1.build(input_shape)
        if self.norm2 is not None:
            self.norm2.build(input_shape)

        # Build attention layers
        self.qkv.build(input_shape)
        self.proj.build(input_shape)

        # Build MLP - Sequential handles its own building
        self.mlp.build(input_shape)

        # Build dropout layers if they exist
        if self.dropout_layer is not None:
            self.dropout_layer.build(input_shape)
        if self.attention_dropout_layer is not None:
            # Attention dropout operates on attention weights, different shape
            # We'll let Keras handle this automatically
            pass

        # Always call parent build at the end
        super().build(input_shape)

    def _apply_masked_attention(
        self,
        attn_weights: keras.KerasTensor,
        mask: Optional[keras.KerasTensor],
        num_patches: int,
        num_queries: int
    ) -> keras.KerasTensor:
        """
        Apply masked attention to query-to-patch interactions.

        Args:
            attn_weights: Attention weights [batch, heads, seq_len, seq_len]
            mask: Mask tensor [batch, num_queries, H, W]
            num_patches: Number of patch tokens
            num_queries: Number of query tokens

        Returns:
            Masked attention weights with same shape as input
        """
        if mask is None or not self.use_masked_attention:
            return attn_weights

        # Apply mask probabilistically during training
        if self.mask_probability < 1.0:
            should_mask = ops.cast(
                keras.random.uniform([]) < self.mask_probability,
                dtype=attn_weights.dtype
            )
            # If should_mask is 0, return original weights
            if ops.all(should_mask == 0):
                return attn_weights

        batch_size = ops.shape(attn_weights)[0]

        # Flatten mask for patch matching: [B, Q, H*W]
        mask_flat = ops.reshape(mask, [batch_size, num_queries, -1])

        # Extract query-to-patch attention: [B, H, Q, P]
        query_to_patch_attn = attn_weights[:, :, num_patches:, :num_patches]

        # Expand mask to match attention heads: [B, 1, Q, P] -> [B, H, Q, P]
        mask_expanded = ops.expand_dims(mask_flat, axis=1)
        mask_expanded = ops.tile(mask_expanded, [1, self.num_heads, 1, 1])

        # Apply mask (set to large negative value where mask is 0)
        large_negative = ops.full_like(query_to_patch_attn, -1e9)
        masked_attn = ops.where(
            mask_expanded > 0.5,
            query_to_patch_attn,
            large_negative
        )

        # Reconstruct full attention weights
        # Top part: patch-to-all (unchanged)
        top_part = attn_weights[:, :, :num_patches, :]

        # Bottom left: query-to-patch (masked)
        bottom_left = masked_attn

        # Bottom right: query-to-query (unchanged)
        bottom_right = attn_weights[:, :, num_patches:, num_patches:]

        # Combine bottom parts
        bottom_part = ops.concatenate([bottom_left, bottom_right], axis=-1)

        # Combine all parts
        attn_weights = ops.concatenate([top_part, bottom_part], axis=-2)

        return attn_weights

    def call(
        self,
        inputs: keras.KerasTensor,
        mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through the EoMT layer.

        Args:
            inputs: Input tensor containing both patch tokens and queries
                   [batch, seq_len, embed_dim] where seq_len = num_patches + num_queries
            mask: Optional mask tensor [batch, num_queries, H, W] for masked attention
            training: Boolean indicating training mode

        Returns:
            Output tensor with same shape as inputs [batch, seq_len, embed_dim]
        """
        batch_size = ops.shape(inputs)[0]
        seq_len = ops.shape(inputs)[1]
        embed_dim = ops.shape(inputs)[2]

        # Determine number of patches and queries
        if mask is not None:
            num_queries = ops.shape(mask)[1]
            num_patches = seq_len - num_queries
        else:
            # Default assumption: no queries, all patches
            num_patches = seq_len
            num_queries = 0

        # Self-attention block
        shortcut = inputs

        # Pre-normalization (if enabled)
        if self.use_layer_norm and self.norm1 is not None:
            x = self.norm1(inputs, training=training)
        else:
            x = inputs

        # Multi-head self-attention
        qkv = self.qkv(x, training=training)
        qkv = ops.reshape(qkv, [batch_size, seq_len, 3, self.num_heads, self.head_dim])
        qkv = ops.transpose(qkv, [2, 0, 3, 1, 4])  # [3, batch, heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention weights
        attn_weights = ops.matmul(q, ops.transpose(k, [0, 1, 3, 2])) * self.scale

        # Apply masked attention if needed
        if self.use_masked_attention and mask is not None and training:
            attn_weights = self._apply_masked_attention(
                attn_weights, mask, num_patches, num_queries
            )

        # Apply softmax to get attention probabilities
        attn_weights = ops.softmax(attn_weights, axis=-1)

        # Apply attention dropout
        if self.attention_dropout_layer is not None:
            attn_weights = self.attention_dropout_layer(attn_weights, training=training)

        # Apply attention to values
        attn_output = ops.matmul(attn_weights, v)  # [batch, heads, seq_len, head_dim]
        attn_output = ops.transpose(attn_output, [0, 2, 1, 3])  # [batch, seq_len, heads, head_dim]
        attn_output = ops.reshape(attn_output, [batch_size, seq_len, embed_dim])

        # Project back to embed_dim
        attn_output = self.proj(attn_output, training=training)

        # Apply dropout to attention output
        if self.dropout_layer is not None:
            attn_output = self.dropout_layer(attn_output, training=training)

        # First residual connection
        x = shortcut + attn_output

        # MLP block
        shortcut = x

        # Pre-normalization for MLP (if enabled)
        if self.use_layer_norm and self.norm2 is not None:
            x = self.norm2(x, training=training)

        # Apply MLP
        x = self.mlp(x, training=training)

        # Second residual connection
        x = shortcut + x

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape (same as input shape)."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "dropout": self.dropout,
            "attention_dropout": self.attention_dropout,
            "use_layer_norm": self.use_layer_norm,
            "activation": activations.serialize(self.activation),
            "use_masked_attention": self.use_masked_attention,
            "mask_probability": self.mask_probability,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
        })
        return config


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class MaskModule(keras.layers.Layer):
    """
    Mask prediction module for Encoder-only Mask Transformer (EoMT).

    This module processes query tokens and pixel-level features to generate both
    class predictions and spatial mask predictions for instance segmentation.
    It implements the dual-head architecture where each query predicts both
    what class it represents and where that class appears spatially.

    **Intent**: Provide the prediction heads for EoMT that convert learned query
    representations into actionable segmentation outputs, bridging between the
    transformer's learned representations and the final task requirements.

    **Architecture**:
    ```
    Query Tokens [batch, num_queries, embed_dim]
           ↓
    ┌─────────────────┬─────────────────┐
    │   Class Head    │   Mask Head     │
    │                 │                 │
    │  Dense(classes) │  MLP(mask_dim)  │
    │       ↓         │       ↓         │
    │ Class Logits    │ Mask Embeddings │
    └─────────────────┴─────────────────┘
                              ↓
    Pixel Features [batch, H, W, embed_dim]
                              ↓
              Dot Product: embeddings @ pixels^T
                              ↓
                   Mask Logits [batch, queries, H, W]

    Final Output: (Class Logits, Mask Logits)
    ```

    **Mathematical Operations**:
    1. **Class Prediction**: cls_logits = Linear(query_tokens)
    2. **Mask Embedding**: mask_emb = MLP(query_tokens)
    3. **Mask Prediction**: mask_logits = mask_emb @ pixel_features^T

    The mask prediction uses learned similarity between query embeddings and
    pixel-level features to determine spatial extent of each predicted class.

    Args:
        num_classes: Integer, number of classes to predict. Must be positive.
            This determines the size of the classification output.
        hidden_dim: Integer, hidden dimension for the mask prediction MLP.
            Controls the capacity of the mask embedding network. Defaults to 256.
        mask_dim: Integer, final dimension of mask embeddings before dot product
            with pixel features. Should match pixel feature dimension for optimal
            performance. Defaults to 256.
        kernel_initializer: String or initializer, weight initialization for
            all dense layers. Defaults to 'glorot_uniform'.
        bias_initializer: String or initializer, bias initialization for
            all dense layers. Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer for dense layer weights.
        bias_regularizer: Optional regularizer for dense layer biases.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        Tuple of two tensors:
        - query_tokens: `(batch_size, num_queries, embed_dim)`
        - pixel_features: `(batch_size, height, width, embed_dim)`

    Output shape:
        Tuple of two tensors:
        - class_logits: `(batch_size, num_queries, num_classes)`
        - mask_logits: `(batch_size, num_queries, height, width)`

    Attributes:
        class_head: Dense layer for class prediction.
        mask_mlp: Sequential MLP for mask embedding generation.

    Example:
        ```python
        # Basic usage for segmentation
        module = MaskModule(num_classes=80, hidden_dim=256, mask_dim=256)

        # Prepare inputs
        query_tokens = keras.Input(shape=(100, 768))  # 100 queries, 768 dim
        pixel_features = keras.Input(shape=(14, 14, 768))  # 14x14 pixel grid

        # Get predictions
        class_logits, mask_logits = module([query_tokens, pixel_features])

        # class_logits: (batch, 100, 80)
        # mask_logits: (batch, 100, 14, 14)

        # Custom configuration
        module = MaskModule(
            num_classes=21,  # PASCAL VOC classes
            hidden_dim=512,  # Larger capacity
            mask_dim=512,
            kernel_regularizer=keras.regularizers.L2(1e-4)
        )

        # In a complete model
        queries = keras.Input(shape=(100, 768))
        pixels = keras.Input(shape=(16, 16, 768))

        cls_pred, mask_pred = module([queries, pixels])

        model = keras.Model([queries, pixels], [cls_pred, mask_pred])
        model.compile(
            optimizer='adamw',
            loss={
                'class_logits': 'sparse_categorical_crossentropy',
                'mask_logits': 'binary_crossentropy'
            }
        )
        ```

    References:
        - "Your ViT is Secretly a Segmentation Model": https://arxiv.org/abs/2312.02113
        - "DETR: End-to-End Object Detection": https://arxiv.org/abs/2005.12872

    Note:
        The mask_dim should typically match the embed_dim of pixel_features for
        optimal dot product computation. The mask prediction creates a similarity
        map between each query and all spatial locations in the image.
    """

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 256,
        mask_dim: int = 256,
        kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
        bias_initializer: Union[str, initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        bias_regularizer: Optional[regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if mask_dim <= 0:
            raise ValueError(f"mask_dim must be positive, got {mask_dim}")

        # Store configuration
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.mask_dim = mask_dim
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        # CREATE all sub-layers in __init__ (following modern Keras 3 pattern)

        # Class prediction head
        self.class_head = layers.Dense(
            self.num_classes,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="class_head"
        )

        # Mask embedding MLP
        self.mask_mlp = keras.Sequential([
            layers.Dense(
                self.hidden_dim,
                activation="relu",
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name="mask_mlp_1"
            ),
            layers.Dense(
                self.hidden_dim,
                activation="relu",
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name="mask_mlp_2"
            ),
            layers.Dense(
                self.mask_dim,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name="mask_mlp_3"
            )
        ], name="mask_mlp")

    def build(self, input_shape: Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]) -> None:
        """
        Build the layer and all its sub-layers.

        Args:
            input_shape: Tuple of (query_shape, pixel_shape)
        """
        query_shape, pixel_shape = input_shape

        # Build class head with query shape
        self.class_head.build(query_shape)

        # Build mask MLP with query shape
        self.mask_mlp.build(query_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: Tuple[keras.KerasTensor, keras.KerasTensor],
        training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Forward pass through the mask module.

        Args:
            inputs: Tuple of (query_tokens, pixel_features)
                - query_tokens: [batch, num_queries, embed_dim]
                - pixel_features: [batch, H, W, embed_dim]
            training: Boolean indicating training mode

        Returns:
            Tuple of (class_logits, mask_logits)
            - class_logits: [batch, num_queries, num_classes]
            - mask_logits: [batch, num_queries, H, W]
        """
        query_tokens, pixel_features = inputs

        # Get shapes
        batch_size = ops.shape(query_tokens)[0]
        num_queries = ops.shape(query_tokens)[1]
        height = ops.shape(pixel_features)[1]
        width = ops.shape(pixel_features)[2]
        embed_dim = ops.shape(pixel_features)[3]

        # Predict class logits
        class_logits = self.class_head(query_tokens, training=training)

        # Predict mask embeddings
        mask_embeddings = self.mask_mlp(query_tokens, training=training)  # [batch, num_queries, mask_dim]

        # Compute mask logits via dot product with pixel features
        # Reshape pixel features for efficient computation
        pixel_features_flat = ops.reshape(
            pixel_features, [batch_size, height * width, embed_dim]
        )  # [batch, H*W, embed_dim]

        # Efficient dot product: [batch, queries, mask_dim] @ [batch, mask_dim, H*W]
        # First transpose pixel features: [batch, H*W, embed_dim] -> [batch, embed_dim, H*W]
        pixel_features_transposed = ops.transpose(pixel_features_flat, [0, 2, 1])

        # Compute dot product: [batch, queries, mask_dim] @ [batch, mask_dim, H*W] -> [batch, queries, H*W]
        mask_logits_flat = ops.matmul(mask_embeddings, pixel_features_transposed)

        # Reshape back to spatial dimensions
        mask_logits = ops.reshape(
            mask_logits_flat, [batch_size, num_queries, height, width]
        )

        return class_logits, mask_logits

    def compute_output_shape(
        self,
        input_shape: Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]
    ) -> Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
        """Compute output shapes."""
        query_shape, pixel_shape = input_shape
        batch_size = query_shape[0]
        num_queries = query_shape[1]
        height, width = pixel_shape[1], pixel_shape[2]

        class_shape = (batch_size, num_queries, self.num_classes)
        mask_shape = (batch_size, num_queries, height, width)

        return class_shape, mask_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "hidden_dim": self.hidden_dim,
            "mask_dim": self.mask_dim,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
        })
        return config


# ---------------------------------------------------------------------