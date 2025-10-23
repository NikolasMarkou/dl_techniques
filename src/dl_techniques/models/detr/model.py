"""
DETR (DEtection TRansformer) Model Implementation in Keras 3

This module provides a Keras 3 implementation of the DETR model, as described in
"End-to-End Object Detection with Transformers" by Carion et al. (2020).

The implementation follows modern Keras 3 best practices and leverages existing
components from the dl_techniques framework including:
- TransformerLayer for encoder/decoder blocks
- Normalization factory for flexible normalization
- FFN factory for configurable feed-forward networks
- Attention factory for attention mechanisms

Key Architectural Features:
---------------------------
- A CNN backbone (e.g., ResNet-50) for feature extraction.
- A 2D sinusoidal positional encoding added to the feature map.
- A standard Transformer architecture with an encoder and a decoder.
- Learned object queries that attend to image features to detect objects.
- Prediction heads (FFNs) for bounding boxes and class labels.

Usage:
------
```python
# Create a DETR model with a ResNet-50 backbone
detr_model = create_detr(
    num_classes=91,
    num_queries=100,
    backbone_name="resnet50"
)

# The model expects preprocessed images and a padding mask
image_input = keras.Input(shape=(None, None, 3), dtype="float32")
mask_input = keras.Input(shape=(None, None), dtype="bool")

outputs = detr_model([image_input, mask_input])

# The model can be saved and loaded seamlessly
detr_model.save("detr_model.keras")
loaded_model = keras.models.load_model("detr_model.keras")
```

Note: The loss function, which involves the Hungarian Matcher for bipartite
matching, is not part of the model architecture itself and is intended to be
implemented within the training loop. This file focuses on the forward-pass
architecture of the DETR model.
"""

import keras
from keras import layers, models
from typing import Optional, Dict, Any, List, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.ffn import create_ffn_layer
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.transformer import TransformerLayer, FFNType, NormalizationType
from dl_techniques.layers.embedding.positional_embedding_sine_2d import PositionEmbeddingSine2D

# ---------------------------------------------------------------------
# Transformer Components
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class DetrTransformer(layers.Layer):
    """
    DETR Transformer combining encoder and decoder stacks.

    **Intent**: To process image features through a transformer architecture,
    allowing the model to capture global dependencies in the image and attend
    to relevant regions for object detection.

    **Architecture**:
    ```
    Input Features + Positional Encoding
           ↓
    Encoder Stack (N Layers)
           ↓
    Memory (Encoded Features)
           ↓
    Decoder Stack (N Layers) ← Object Queries + Query Positional Encoding
           ↓
    Output Features (for each query)
    ```

    Args:
        hidden_dim: The dimensionality of the transformer. Defaults to 256.
        num_heads: The number of attention heads. Defaults to 8.
        num_encoder_layers: Number of encoder layers. Defaults to 6.
        num_decoder_layers: Number of decoder layers. Defaults to 6.
        ffn_dim: The hidden dimension of the FFN. Defaults to 2048.
        dropout: The dropout rate. Defaults to 0.1.
        activation: Activation function for FFN. Defaults to "relu".
        normalization_type: Type of normalization. Defaults to "layer_norm".
        ffn_type: Type of FFN to use. Defaults to "mlp".
        **kwargs: Additional layer arguments.

    Input shape:
        Tuple of:
        - src: `(batch_size, H*W, hidden_dim)` - Flattened image features
        - mask: `(batch_size, H*W)` - Padding mask
        - query_embed: `(num_queries, hidden_dim)` - Object query embeddings
        - pos_embed: `(batch_size, H*W, hidden_dim)` - Positional encodings

    Output shape:
        List of `(batch_size, num_queries, hidden_dim)` tensors, one for each decoder layer
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        normalization_type: NormalizationType = "layer_norm",
        ffn_type: FFNType = "mlp",
        **kwargs
    ):
        super().__init__(**kwargs)

        if hidden_dim <= 0 or num_heads <= 0 or ffn_dim <= 0:
            raise ValueError("Dimensions must be positive.")
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads}).")
        if num_encoder_layers <= 0 or num_decoder_layers <= 0:
            raise ValueError("Number of layers must be positive.")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.ffn_dim = ffn_dim
        self.dropout_rate = dropout
        self.activation = activation
        self.normalization_type = normalization_type
        self.ffn_type = ffn_type

        # Create encoder layers using TransformerLayer
        self.encoder_layers = []
        for i in range(num_encoder_layers):
            self.encoder_layers.append(
                TransformerLayer(
                    hidden_size=hidden_dim,
                    num_heads=num_heads,
                    intermediate_size=ffn_dim,
                    dropout_rate=dropout,
                    activation=activation,
                    normalization_type=normalization_type,
                    normalization_position='pre',
                    ffn_type=ffn_type,
                    attention_type='multi_head_attention',
                    name=f"encoder_layer_{i}"
                )
            )

        # Create decoder layers
        self.decoder_layers = []
        for i in range(num_decoder_layers):
            self.decoder_layers.append(
                DetrDecoderLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    activation=activation,
                    normalization_type=normalization_type,
                    ffn_type=ffn_type,
                    name=f"decoder_layer_{i}"
                )
            )

    def build(self, input_shape: Tuple[Tuple[int, ...], ...]) -> None:
        """Build encoder and decoder layers."""
        src_shape, mask_shape, query_embed_shape, pos_embed_shape = input_shape

        # Build encoder layers
        for encoder_layer in self.encoder_layers:
            encoder_layer.build(src_shape)

        # Build decoder layers
        # Decoder input shape is (batch_size, num_queries, hidden_dim)
        decoder_input_shape = (src_shape[0], query_embed_shape[0], self.hidden_dim)
        for decoder_layer in self.decoder_layers:
            decoder_layer.build((decoder_input_shape, src_shape))

        super().build(input_shape)

    def call(
        self,
        src: keras.KerasTensor,
        mask: keras.KerasTensor,
        query_embed: keras.KerasTensor,
        pos_embed: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> List[keras.KerasTensor]:
        """
        Forward pass through encoder and decoder.

        Args:
            src: Source features (batch_size, H*W, hidden_dim)
            mask: Padding mask (batch_size, H*W)
            query_embed: Object query embeddings (num_queries, hidden_dim)
            pos_embed: Positional encodings (batch_size, H*W, hidden_dim)
            training: Training mode flag

        Returns:
            List of decoder outputs, one per layer
        """
        # Encoder forward pass
        memory = src
        for encoder_layer in self.encoder_layers:
            # Add positional encoding to the input for each encoder layer
            memory = encoder_layer(memory + pos_embed, training=training)

        # Decoder forward pass
        # Initialize target with zeros and add query embeddings
        batch_size = keras.ops.shape(src)[0]
        num_queries = keras.ops.shape(query_embed)[0]

        # Expand query_embed to batch dimension
        tgt = keras.ops.zeros((batch_size, num_queries, self.hidden_dim))

        decoder_outputs = []
        for decoder_layer in self.decoder_layers:
            tgt = decoder_layer(
                tgt=tgt,
                memory=memory,
                query_pos=query_embed,
                pos_embed=pos_embed,
                training=training
            )
            decoder_outputs.append(tgt)

        return decoder_outputs

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "num_encoder_layers": self.num_encoder_layers,
            "num_decoder_layers": self.num_decoder_layers,
            "ffn_dim": self.ffn_dim,
            "dropout": self.dropout_rate,
            "activation": self.activation,
            "normalization_type": self.normalization_type,
            "ffn_type": self.ffn_type,
        })
        return config


@keras.saving.register_keras_serializable()
class DetrDecoderLayer(layers.Layer):
    """
    A single DETR Transformer Decoder Layer with pre-normalization.

    **Intent**: To produce object-centric features by attending to the encoder's
    output (`memory`) using a set of learned `object queries`. It uses two
    attention mechanisms: self-attention among queries and cross-attention
    between queries and image features.

    **Architecture**:
    ```
    Input (tgt) & Memory
         ↓ (tgt)
    Norm1 → Self-Attention → Add with Input (tgt)
         ↓
    Norm2 → Cross-Attention(Q=tgt, K,V=memory) → Add
         ↓
    Norm3 → FFN → Add with previous output
         ↓
       Output
    ```

    Args:
        hidden_dim: The dimensionality of features.
        num_heads: The number of attention heads.
        ffn_dim: The hidden dimension of the feed-forward network.
        dropout: The dropout rate. Defaults to 0.1.
        activation: The activation for the FFN. Defaults to "relu".
        normalization_type: Type of normalization. Defaults to "layer_norm".
        ffn_type: Type of FFN to use. Defaults to "mlp".
        **kwargs: Additional layer arguments.

    Input shape:
        - tgt: `(batch_size, num_queries, hidden_dim)`
        - memory: `(batch_size, H*W, hidden_dim)`
        - query_pos: `(num_queries, hidden_dim)` (optional)
        - pos_embed: `(batch_size, H*W, hidden_dim)` (optional)

    Output shape:
        - `(batch_size, num_queries, hidden_dim)`
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        activation: str = "relu",
        normalization_type: str = "layer_norm",
        ffn_type: str = "mlp",
        **kwargs
    ):
        super().__init__(**kwargs)

        if hidden_dim <= 0 or num_heads <= 0 or ffn_dim <= 0:
            raise ValueError("Dimensions must be positive.")
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads}).")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout_rate = dropout
        self.activation = activation
        self.normalization_type = normalization_type
        self.ffn_type = ffn_type

        # Self-attention
        self.self_attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=hidden_dim // num_heads,
            dropout=dropout,
            name="self_attn"
        )

        # Cross-attention
        self.cross_attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=hidden_dim // num_heads,
            dropout=dropout,
            name="cross_attn"
        )

        # Feed-forward network using factory
        self.ffn = create_ffn_layer(
            ffn_type,
            hidden_dim=ffn_dim,
            output_dim=hidden_dim,
            activation=activation,
            dropout_rate=dropout,
            name="ffn"
        )

        # Normalization layers using factory
        self.norm1 = create_normalization_layer(normalization_type, name="norm1")
        self.norm2 = create_normalization_layer(normalization_type, name="norm2")
        self.norm3 = create_normalization_layer(normalization_type, name="norm3")

        # Dropout layers
        self.dropout1 = layers.Dropout(dropout, name="dropout1")
        self.dropout2 = layers.Dropout(dropout, name="dropout2")
        self.dropout3 = layers.Dropout(dropout, name="dropout3")

    def build(self, input_shape: Tuple[Tuple[int, ...], Tuple[int, ...]]) -> None:
        """Build all sub-layers."""
        tgt_shape, memory_shape = input_shape

        # Build normalization layers
        self.norm1.build(tgt_shape)
        self.norm2.build(tgt_shape)
        self.norm3.build(tgt_shape)

        # Build FFN
        self.ffn.build(tgt_shape)

        super().build(input_shape)

    def call(
        self,
        tgt: keras.KerasTensor,
        memory: keras.KerasTensor,
        query_pos: Optional[keras.KerasTensor] = None,
        pos_embed: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through decoder layer.

        Args:
            tgt: Target queries (batch_size, num_queries, hidden_dim)
            memory: Encoder memory (batch_size, H*W, hidden_dim)
            query_pos: Query positional encodings (num_queries, hidden_dim)
            pos_embed: Positional encodings for memory (batch_size, H*W, hidden_dim)
            training: Training mode flag

        Returns:
            Updated target queries
        """
        # Self-attention block
        tgt_norm = self.norm1(tgt, training=training)

        # Add query positional encoding to query and key for self-attention
        if query_pos is not None:
            # Expand query_pos to batch dimension
            batch_size = keras.ops.shape(tgt)[0]
            query_pos_expanded = keras.ops.tile(
                keras.ops.expand_dims(query_pos, axis=0),
                [batch_size, 1, 1]
            )
            q = k = tgt_norm + query_pos_expanded
        else:
            q = k = tgt_norm

        attn_output = self.self_attn(
            query=q,
            value=tgt_norm,
            key=k,
            training=training
        )
        tgt = tgt + self.dropout1(attn_output, training=training)

        # Cross-attention block
        tgt_norm = self.norm2(tgt, training=training)

        # Add query positional encoding to query
        if query_pos is not None:
            q = tgt_norm + query_pos_expanded
        else:
            q = tgt_norm

        # Add positional encoding to memory for key
        if pos_embed is not None:
            k = memory + pos_embed
        else:
            k = memory

        cross_attn_output = self.cross_attn(
            query=q,
            value=memory,
            key=k,
            training=training
        )
        tgt = tgt + self.dropout2(cross_attn_output, training=training)

        # FFN block
        tgt_norm = self.norm3(tgt, training=training)
        ffn_output = self.ffn(tgt_norm, training=training)
        tgt = tgt + self.dropout3(ffn_output, training=training)

        return tgt

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "ffn_dim": self.ffn_dim,
            "dropout": self.dropout_rate,
            "activation": self.activation,
            "normalization_type": self.normalization_type,
            "ffn_type": self.ffn_type,
        })
        return config


# ---------------------------------------------------------------------
# Main DETR Model
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class DETR(models.Model):
    """
    The complete DETR model for end-to-end object detection.

    **Intent**: To provide a complete, end-to-end model that takes an image
    and returns a set of object detections (class probabilities and bounding boxes).

    **Architecture**:
    ```
    Image → Backbone(CNN) → 1x1 Conv ↘
                                     + → Transformer → Prediction Heads → Detections
    Mask  → PositionalEncoding(Sine) ↗
    ```

    Args:
        num_classes: Number of object classes (excluding "no object").
        num_queries: Max number of detections per image.
        backbone: A Keras model (CNN) for feature extraction.
        transformer: The DETR transformer module.
        hidden_dim: Dimensionality of the transformer. Defaults to 256.
        aux_loss: If True, returns predictions from all intermediate decoder
            layers for auxiliary loss calculation. Defaults to True.
        **kwargs: Additional model arguments.

    Input shape:
        Tuple of:
        - images: `(batch_size, height, width, 3)`
        - padding_mask: `(batch_size, height, width)` boolean mask

    Output shape:
        Dictionary containing:
        - pred_logits: `(batch_size, num_queries, num_classes + 1)`
        - pred_boxes: `(batch_size, num_queries, 4)`
        - aux_outputs: List of dicts (if aux_loss=True)
    """

    def __init__(
        self,
        num_classes: int,
        num_queries: int,
        backbone: models.Model,
        transformer: DetrTransformer,
        hidden_dim: int = 256,
        aux_loss: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        if num_classes <= 0 or num_queries <= 0 or hidden_dim <= 0:
            raise ValueError("num_classes, num_queries, and hidden_dim must be positive.")

        self.num_classes = num_classes
        self.num_queries = num_queries
        self.backbone = backbone
        self.transformer = transformer
        self.hidden_dim = hidden_dim
        self.aux_loss = aux_loss

        # Prediction heads
        self.class_embed = layers.Dense(num_classes + 1, name="class_embed")

        # Bbox prediction head using FFN factory
        # Note: MLP with 3 layers for bbox prediction
        self.bbox_embed = create_ffn_layer(
            'mlp',
            hidden_dim=hidden_dim,
            output_dim=4,
            activation='relu',
            dropout_rate=0.0,
            name="bbox_embed"
        )

        # Query embeddings
        self.query_embed = layers.Embedding(num_queries, hidden_dim, name="query_embed")

        # Input projection
        self.input_proj = layers.Conv2D(hidden_dim, kernel_size=1, name="input_proj")

        # Positional embedding
        self.pos_embed = PositionEmbeddingSine2D(num_pos_feats=hidden_dim // 2, name="pos_embed")

    def call(
        self,
        inputs: Tuple[keras.KerasTensor, keras.KerasTensor],
        training: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Forward pass through DETR model.

        Args:
            inputs: Tuple of (images, padding_mask)
            training: Training mode flag

        Returns:
            Dictionary with predictions
        """
        images, padding_mask = inputs

        # Extract features from backbone
        features = self.backbone(images, training=training)

        # Generate positional encodings
        pos_embed = self.pos_embed(features, mask=padding_mask)

        # Project features to transformer dimension
        projected_features = self.input_proj(features)

        # Flatten spatial dimensions: (B, H, W, C) -> (B, H*W, C)
        batch_size = keras.ops.shape(projected_features)[0]
        height = keras.ops.shape(projected_features)[1]
        width = keras.ops.shape(projected_features)[2]

        src = keras.ops.reshape(projected_features, (batch_size, height * width, self.hidden_dim))
        pos_embed_flat = keras.ops.reshape(pos_embed, (batch_size, height * width, self.hidden_dim))

        # Flatten mask if provided
        if padding_mask is not None:
            mask_flat = keras.ops.reshape(padding_mask, (batch_size, height * width))
        else:
            mask_flat = None

        # Get query embeddings - this gives us the embedding matrix
        query_embed_weights = self.query_embed.weights[0]

        # Pass through transformer
        hs = self.transformer(src, mask_flat, query_embed_weights, pos_embed_flat, training=training)

        # Apply prediction heads to all decoder outputs
        outputs_class = [self.class_embed(h) for h in hs]
        outputs_coord = [keras.ops.sigmoid(self.bbox_embed(h)) for h in hs]

        # Prepare output dictionary
        last_output = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
        }

        # Add auxiliary outputs if requested
        if self.aux_loss:
            aux_outputs = [
                {"pred_logits": a, "pred_boxes": b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
            ]
            last_output["aux_outputs"] = aux_outputs

        return last_output

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "num_queries": self.num_queries,
            "hidden_dim": self.hidden_dim,
            "aux_loss": self.aux_loss,
            "backbone": keras.saving.serialize_keras_object(self.backbone),
            "transformer": keras.saving.serialize_keras_object(self.transformer),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DETR":
        """Deserialize model from configuration."""
        backbone_config = config.pop("backbone")
        transformer_config = config.pop("transformer")
        backbone = keras.saving.deserialize_keras_object(backbone_config)
        transformer = keras.saving.deserialize_keras_object(transformer_config)
        return cls(backbone=backbone, transformer=transformer, **config)


# ---------------------------------------------------------------------
# Factory Function
# ---------------------------------------------------------------------


def create_detr(
    num_classes: int,
    num_queries: int,
    backbone_name: str = "resnet50",
    backbone_trainable: bool = False,
    hidden_dim: int = 256,
    num_heads: int = 8,
    num_encoder_layers: int = 6,
    num_decoder_layers: int = 6,
    ffn_dim: int = 2048,
    dropout: float = 0.1,
    aux_loss: bool = True,
    activation: str = "relu",
    normalization_type: str = "layer_norm",
    ffn_type: str = "mlp"
) -> DETR:
    """
    Convenience factory to create a DETR model with a specified backbone.

    Args:
        num_classes: Number of object detection classes.
        num_queries: Number of object queries.
        backbone_name: Name of the CNN backbone ("resnet50").
        backbone_trainable: If True, the backbone weights will be fine-tuned.
        hidden_dim: Dimensionality of the transformer.
        num_heads: Number of attention heads.
        num_encoder_layers: Number of encoder layers.
        num_decoder_layers: Number of decoder layers.
        ffn_dim: Hidden dimension of the FFNs in the transformer.
        dropout: Dropout rate used in the transformer.
        aux_loss: If True, model outputs predictions from intermediate layers.
        activation: Activation function for FFN. Defaults to "relu".
        normalization_type: Type of normalization to use. Defaults to "layer_norm".
        ffn_type: Type of FFN to use. Defaults to "mlp".

    Returns:
        A DETR Keras model instance.
    """
    if backbone_name == "resnet50":
        base_model = keras.applications.ResNet50(
            include_top=False, weights="imagenet", input_shape=None
        )
        feature_layer_name = "conv4_block6_out"
        backbone_model = models.Model(
            inputs=base_model.input,
            outputs=base_model.get_layer(feature_layer_name).output,
            name=f"{backbone_name}_backbone"
        )
    else:
        raise NotImplementedError(f"Backbone '{backbone_name}' not supported.")

    backbone_model.trainable = backbone_trainable
    logger.info(f"Created backbone '{backbone_name}' with trainable={backbone_trainable}")

    transformer = DetrTransformer(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        ffn_dim=ffn_dim,
        dropout=dropout,
        activation=activation,
        normalization_type=normalization_type,
        ffn_type=ffn_type
    )

    detr_model = DETR(
        num_classes=num_classes,
        num_queries=num_queries,
        backbone=backbone_model,
        transformer=transformer,
        hidden_dim=hidden_dim,
        aux_loss=aux_loss
    )

    logger.info(f"Created DETR model with {num_queries} queries for {num_classes} classes.")
    return detr_model