"""
DETR (DEtection TRansformer) Model Implementation in Keras 3

This module provides a Keras 3 implementation of the DETR model, as described in
"End-to-End Object Detection with Transformers" by Carion et al. (2020).

The implementation follows modern Keras 3 best practices and leverages existing
components from the dl_techniques framework including:
- TransformerLayer for encoder blocks
- TransformerDecoderLayer for decoder blocks
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
from dl_techniques.layers.transformers import TransformerLayer, TransformerDecoderLayer, FFNType, NormalizationType
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
                    attention_type='multi_head',  # fix 1a: was 'multi_head_attention'
                    name=f"encoder_layer_{i}"
                )
            )

        # Create decoder layers using TransformerDecoderLayer (fix 1c/1d: replaced DetrDecoderLayer)
        self.decoder_layers = []
        for i in range(num_decoder_layers):
            self.decoder_layers.append(
                TransformerDecoderLayer(
                    hidden_size=hidden_dim,
                    num_heads=num_heads,
                    intermediate_size=ffn_dim,
                    dropout_rate=dropout,
                    activation=activation,
                    normalization_type=normalization_type,
                    normalization_position='pre',
                    use_causal_mask=False,
                    ffn_type=ffn_type,
                    name=f"decoder_layer_{i}"
                )
            )

    def build(self, input_shape) -> None:
        """Explicitly build all encoder and decoder sub-layers.

        Required for .keras round-trip: Keras weight-restore needs every
        sub-layer to be built before weights can be re-indexed by path.
        Lazy build (super().build() only) leaves sub-layers unbuilt at load
        time, causing a weight count mismatch.
        """
        enc_shape = (None, None, self.hidden_dim)
        dec_shape = (None, None, self.hidden_dim)
        for layer in self.encoder_layers:
            if not layer.built:
                layer.build(enc_shape)
        for layer in self.decoder_layers:
            if not layer.built:
                layer.build(dec_shape)
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
            mask: Padding mask (batch_size, H*W) -- currently unused (see fix 1i)
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

        # Decoder forward pass (fix 1f)
        batch_size = keras.ops.shape(src)[0]
        num_queries = keras.ops.shape(query_embed)[0]
        query_embed_expanded = keras.ops.tile(
            keras.ops.expand_dims(query_embed, axis=0),
            [batch_size, 1, 1]
        )
        tgt = keras.ops.zeros((batch_size, num_queries, self.hidden_dim))

        decoder_outputs = []
        for decoder_layer in self.decoder_layers:
            tgt = decoder_layer(
                tgt + query_embed_expanded,
                memory,
                training=training
            )
            decoder_outputs.append(tgt)

        return decoder_outputs

    def compute_output_shape(self, input_shape):
        """Output is a list of per-decoder-layer query feature shapes."""
        src_shape, _, query_embed_shape, _ = input_shape
        single = (src_shape[0], query_embed_shape[0], self.hidden_dim)
        return [single] * self.num_decoder_layers

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

    def build(self, input_shape) -> None:
        """Build DETR and all sublayers so .keras round-trip can restore weights.

        Keras weight-restore calls load_own_variables() on each sublayer; if a
        sublayer is unbuilt (no variables), load_own_variables raises when the
        saved store has more entries than the (empty) variable list.  Building
        here ensures every sublayer has its variables before weights are loaded.
        """
        # input_shape is either [(B,H,W,3),(B,H,W)] or (B,H,W,3)
        if isinstance(input_shape, (list, tuple)) and len(input_shape) == 2:
            images_shape = input_shape[0]
        else:
            images_shape = input_shape

        if not self.backbone.built:
            self.backbone.build(images_shape)

        backbone_out_shape = self.backbone.compute_output_shape(images_shape)
        if not self.input_proj.built:
            self.input_proj.build(backbone_out_shape)

        seq_shape = (None, None, self.hidden_dim)
        if not self.class_embed.built:
            self.class_embed.build(seq_shape)
        if not self.bbox_embed.built:
            self.bbox_embed.build(seq_shape)
        if not self.query_embed.built:
            self.query_embed.build((None,))

        if not self.transformer.built:
            self.transformer.build(seq_shape)

        super().build(input_shape)

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

        # Project features to transformer dimension first so pos_embed is
        # computed at feature-map resolution, not full-image resolution.
        # padding_mask is (B, img_H, img_W); PositionEmbeddingSine2D uses the
        # mask's spatial dims to build the encoding grid, so passing the
        # full-image mask would produce (B, C, img_H, img_W) -- wrong size.
        # We pass mask=None so the layer uses projected_features' spatial dims.
        projected_features = self.input_proj(features)

        # fix 1g: compute pos_embed on projected_features (feature-map resolution).
        # PositionEmbeddingSine2D returns (B, C, H, W) channels-first;
        # transpose to (B, H, W, C) to match projected_features layout.
        pos_embed = self.pos_embed(projected_features)
        pos_embed = keras.ops.transpose(pos_embed, [0, 2, 3, 1])

        # Flatten spatial dimensions: (B, H, W, C) -> (B, H*W, C)
        batch_size = keras.ops.shape(projected_features)[0]
        height = keras.ops.shape(projected_features)[1]
        width = keras.ops.shape(projected_features)[2]

        src = keras.ops.reshape(projected_features, (batch_size, height * width, self.hidden_dim))
        pos_embed_flat = keras.ops.reshape(pos_embed, (batch_size, height * width, self.hidden_dim))

        # fix 1h: get query embeddings via a forward call (build-safe; avoids
        # accessing .embeddings before the Embedding layer is built).
        query_embed_weights = self.query_embed(keras.ops.arange(self.num_queries))

        # fix 1i: mask_flat is (B, H*W); TransformerLayer expects (B,T,T) attention_mask shape.
        # Masking support deferred -- pass None for now.
        hs = self.transformer(src, None, query_embed_weights, pos_embed_flat, training=training)

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
        """Deserialize model from configuration. fix 1k: defensive copy."""
        config = dict(config)
        backbone = keras.saving.deserialize_keras_object(config.pop("backbone"))
        transformer = keras.saving.deserialize_keras_object(config.pop("transformer"))
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
