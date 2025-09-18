"""
DETR (DEtection TRansformer) Model Implementation in Keras 3

This module provides a Keras 3 implementation of the DETR model, as described in
"End-to-End Object Detection with Transformers" by Carion et al. (2020).

The implementation has been structured following modern Keras 3 best practices
to ensure robustness, clarity, and proper serialization.

Key Architectural Features:
---------------------------
- A CNN backbone (e.g., ResNet-50) for feature extraction.
- A 2D sinusoidal positional encoding added to the feature map.
- A standard Transformer architecture with an encoder and a decoder.
- Learned object queries that attend to image features to detect objects.
- Prediction heads (FFNs) for bounding boxes and class labels.

Usage:
------```python
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
loaded_model = keras.models.load_model("detr_model.keras")```

Note: The loss function, which involves the Hungarian Matcher for bipartite
matching, is not part of the model architecture itself and is intended to be
implemented within the training loop. This file focuses on the forward-pass
architecture of the DETR model.
"""

import keras
import math
from keras import layers, models, activations
from typing import Optional, Dict, Any, List, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
# Assuming the rewritten PositionEmbeddingSine2D is available at this path
from dl_techniques.layers.embedding.positional_embedding_sine_2d import PositionEmbeddingSine2D


# ---------------------------------------------------------------------
# Transformer Components
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class DetrTransformerEncoderLayer(layers.Layer):
    """
    A single DETR Transformer Encoder Layer with pre-normalization.

    **Intent**: To process and refine image features by applying self-attention,
    allowing different spatial locations to interact and exchange information.
    Positional encodings are added to the query and key to retain spatial awareness.

    **Architecture**:
    ```
        Input (src)
           ↓
    LayerNorm → (+) → MultiHeadAttention → (+) → Dropout → Add with Input
           ↓      ↑                        ↓
        pos_embed                      Residual
           ↓
    LayerNorm → FFN(Linear→ReLU→Dropout→Linear) → Dropout → Add with previous output
           ↓
        Output
    ```

    Args:
        hidden_dim: The dimensionality of the input and output features.
        num_heads: The number of attention heads in the MultiHeadAttention layer.
        ffn_dim: The hidden dimension of the feed-forward network.
        dropout: The dropout rate for regularization. Defaults to 0.1.
        activation: The activation function for the FFN. Defaults to "relu".
        **kwargs: Additional layer arguments.

    Input shape:
        - src: `(batch_size, sequence_length, hidden_dim)`
        - pos: `(batch_size, sequence_length, hidden_dim)`
        - src_key_padding_mask: `(batch_size, sequence_length)` (optional)

    Output shape:
        - `(batch_size, sequence_length, hidden_dim)`
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        activation: str = "relu",
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
        self.activation = activations.get(activation)

        self.self_attn = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=hidden_dim // num_heads, name="self_attn"
        )
        self.ffn = models.Sequential([
            layers.Dense(ffn_dim, activation=self.activation, name="ffn_dense1"),
            layers.Dropout(dropout, name="ffn_dropout"),
            layers.Dense(hidden_dim, name="ffn_dense2")
        ], name="ffn")

        self.norm1 = layers.LayerNormalization(name="norm1")
        self.norm2 = layers.LayerNormalization(name="norm2")
        self.dropout1 = layers.Dropout(dropout, name="dropout1")
        self.dropout2 = layers.Dropout(dropout, name="dropout2")

    def build(self, input_shape: Tuple[Tuple[int, ...], ...]) -> None:
        src_shape, _, _ = input_shape
        self.norm1.build(src_shape)
        self.norm2.build(src_shape)
        self.ffn.build(src_shape)
        super().build(input_shape)

    def call(self, src: keras.KerasTensor, pos: keras.KerasTensor,
             src_key_padding_mask: Optional[keras.KerasTensor] = None) -> keras.KerasTensor:
        src_norm = self.norm1(src)
        q = k = src_norm + pos

        attn_output = self.self_attn(
            query=q, value=src_norm, key=k, attention_mask=src_key_padding_mask
        )
        src = src + self.dropout1(attn_output)

        src2 = self.norm2(src)
        ffn_output = self.ffn(src2)
        src = src + self.dropout2(ffn_output)
        return src

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "ffn_dim": self.ffn_dim,
            "dropout": self.dropout_rate,
            "activation": activations.serialize(self.activation)
        })
        return config


@keras.saving.register_keras_serializable()
class DetrTransformerDecoderLayer(layers.Layer):
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
    LayerNorm → (+) → Self-Attention → (+) → Dropout → Add with Input (tgt)
         ↓      ↑                      ↓
    query_pos                    Residual
         ↓
    LayerNorm → (+) → Cross-Attention(Q=tgt, K,V=memory) → (+) → Dropout → Add
         ↓      ↑                      ↓
    query_pos                    Residual
         ↓
    LayerNorm → FFN(Linear→ReLU→Dropout→Linear) → Dropout → Add with previous output
         ↓
       Output
    ```

    Args:
        hidden_dim: The dimensionality of features.
        num_heads: The number of attention heads.
        ffn_dim: The hidden dimension of the feed-forward network.
        dropout: The dropout rate. Defaults to 0.1.
        activation: The activation for the FFN. Defaults to "relu".
        **kwargs: Additional layer arguments.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        activation: str = "relu",
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
        self.activation = activations.get(activation)

        self.self_attn = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=hidden_dim // num_heads, name="self_attn"
        )
        self.cross_attn = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=hidden_dim // num_heads, name="cross_attn"
        )
        self.ffn = models.Sequential([
            layers.Dense(ffn_dim, activation=self.activation, name="ffn_dense1"),
            layers.Dropout(dropout, name="ffn_dropout"),
            layers.Dense(hidden_dim, name="ffn_dense2")
        ], name="ffn")

        self.norm1 = layers.LayerNormalization(name="norm1")
        self.norm2 = layers.LayerNormalization(name="norm2")
        self.norm3 = layers.LayerNormalization(name="norm3")
        self.dropout1 = layers.Dropout(dropout, name="dropout1")
        self.dropout2 = layers.Dropout(dropout, name="dropout2")
        self.dropout3 = layers.Dropout(dropout, name="dropout3")

    def build(self, input_shape: Tuple[Tuple[int, ...], ...]) -> None:
        tgt_shape, memory_shape, _, _, _ = input_shape
        self.norm1.build(tgt_shape)
        self.norm2.build(tgt_shape)
        self.norm3.build(tgt_shape)
        self.ffn.build(tgt_shape)
        super().build(input_shape)

    def call(
        self,
        tgt: keras.KerasTensor,
        memory: keras.KerasTensor,
        query_pos: keras.KerasTensor,
        pos: keras.KerasTensor,
        memory_key_padding_mask: Optional[keras.KerasTensor] = None
    ) -> keras.KerasTensor:
        tgt_norm = self.norm1(tgt)
        q = k = tgt_norm + query_pos
        self_attn_output = self.self_attn(query=q, key=k, value=tgt_norm)
        tgt = tgt + self.dropout1(self_attn_output)

        tgt_norm2 = self.norm2(tgt)
        cross_attn_output = self.cross_attn(
            query=tgt_norm2 + query_pos,
            key=memory + pos,
            value=memory,
            attention_mask=memory_key_padding_mask
        )
        tgt = tgt + self.dropout2(cross_attn_output)

        tgt_norm3 = self.norm3(tgt)
        ffn_output = self.ffn(tgt_norm3)
        tgt = tgt + self.dropout3(ffn_output)
        return tgt

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "ffn_dim": self.ffn_dim,
            "dropout": self.dropout_rate,
            "activation": activations.serialize(self.activation)
        })
        return config


@keras.saving.register_keras_serializable()
class DetrTransformer(layers.Layer):
    """
    The main DETR Transformer module, containing encoder and decoder stacks.

    **Intent**: To transform image features into a set of object-centric embeddings.
    It first enriches image features with self-attention (encoder) and then uses
    object queries to extract information about potential objects (decoder).

    Args:
        hidden_dim: Dimensionality of the transformer. Defaults to 256.
        num_heads: Number of attention heads. Defaults to 8.
        num_encoder_layers: Number of encoder layers. Defaults to 6.
        num_decoder_layers: Number of decoder layers. Defaults to 6.
        ffn_dim: Hidden dimension of FFNs. Defaults to 2048.
        dropout: Dropout rate. Defaults to 0.1.
        **kwargs: Additional layer arguments.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.ffn_dim = ffn_dim
        self.dropout = dropout

        self.encoder_layers = [
            DetrTransformerEncoderLayer(hidden_dim, num_heads, ffn_dim, dropout, name=f"encoder_layer_{i}")
            for i in range(num_encoder_layers)
        ]
        self.decoder_layers = [
            DetrTransformerDecoderLayer(hidden_dim, num_heads, ffn_dim, dropout, name=f"decoder_layer_{i}")
            for i in range(num_decoder_layers)
        ]
        self.encoder_norm = layers.LayerNormalization(name="encoder_norm")
        self.decoder_norm = layers.LayerNormalization(name="decoder_norm")

    def build(self, input_shape: Tuple[Tuple[int, ...], ...]) -> None:
        src_shape, _, query_embed_shape, pos_embed_shape = input_shape
        bs, h, w, c = src_shape
        flat_shape = (bs, h * w if h is not None and w is not None else None, c)

        for layer in self.encoder_layers:
            layer.build((flat_shape, pos_embed_shape, (bs, h * w)))
        self.encoder_norm.build(flat_shape)

        tgt_shape = (bs, query_embed_shape[0], query_embed_shape[1])
        for layer in self.decoder_layers:
            layer.build((tgt_shape, flat_shape, tgt_shape, pos_embed_shape, (bs, h * w)))
        self.decoder_norm.build(tgt_shape)

        super().build(input_shape)

    def call(self, src, mask, query_embed, pos_embed):
        bs, h, w, c = keras.ops.shape(src)
        src = keras.ops.reshape(src, (bs, -1, c))
        mask = keras.ops.reshape(mask, (bs, -1))
        pos_embed = keras.ops.reshape(pos_embed, (bs, -1, c))

        query_embed = keras.ops.expand_dims(query_embed, axis=0)
        query_embed = keras.ops.tile(query_embed, [bs, 1, 1])

        memory = src
        for layer in self.encoder_layers:
            memory = layer(memory, pos_embed, src_key_padding_mask=mask)
        memory = self.encoder_norm(memory)

        tgt = keras.ops.zeros_like(query_embed)
        intermediate_outputs = []
        for layer in self.decoder_layers:
            tgt = layer(
                tgt, memory, query_pos=query_embed, pos=pos_embed, memory_key_padding_mask=mask
            )
            intermediate_outputs.append(self.decoder_norm(tgt))

        return keras.ops.stack(intermediate_outputs)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "num_encoder_layers": self.num_encoder_layers,
            "num_decoder_layers": self.num_decoder_layers,
            "ffn_dim": self.ffn_dim,
            "dropout": self.dropout,
        })
        return config


# ---------------------------------------------------------------------
# Prediction Heads
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class MLP(layers.Layer):
    """
    A simple multi-layer perceptron (FFN).

    **Intent**: To act as a prediction head, transforming the transformer's
    output embeddings into class logits or bounding box coordinates.

    **Architecture**:
    `Input → [Dense(hidden_dim) → ReLU] * (num_layers - 1) → Dense(output_dim) → Output`

    Args:
        hidden_dim: The dimensionality of the hidden layers.
        output_dim: The final output dimension.
        num_layers: The total number of dense layers. Must be at least 1.
        **kwargs: Additional layer arguments.
    """

    def __init__(self, hidden_dim: int, output_dim: int, num_layers: int, **kwargs):
        super().__init__(**kwargs)
        if hidden_dim <= 0 or output_dim <= 0 or num_layers <= 0:
            raise ValueError("Dimensions and num_layers must be positive.")

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.dense_layers = []
        for i in range(num_layers - 1):
            self.dense_layers.append(layers.Dense(hidden_dim, activation="relu", name=f"dense_{i}"))
        self.dense_layers.append(layers.Dense(output_dim, name=f"dense_{num_layers - 1}"))
        # Use a Sequential model for easier building
        self.model = models.Sequential(self.dense_layers, name="mlp_sequential")

    def build(self, input_shape: Tuple[int, ...]) -> None:
        self.model.build(input_shape)
        super().build(input_shape)

    def call(self, x: keras.KerasTensor) -> keras.KerasTensor:
        return self.model(x)

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
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

        self.class_embed = layers.Dense(num_classes + 1, name="class_embed")
        self.bbox_embed = MLP(hidden_dim, 4, 3, name="bbox_embed")
        self.query_embed = layers.Embedding(num_queries, hidden_dim, name="query_embed")
        self.input_proj = layers.Conv2D(hidden_dim, kernel_size=1, name="input_proj")
        self.pos_embed = PositionEmbeddingSine2D(num_pos_feats=hidden_dim // 2, name="pos_embed")

    def call(self, inputs: Tuple[keras.KerasTensor, keras.KerasTensor], training: Optional[bool] = None) -> Dict[str, Any]:
        images, padding_mask = inputs

        features = self.backbone(images, training=training)
        pos_embed = self.pos_embed(features, mask=padding_mask)
        projected_features = self.input_proj(features)

        # self.query_embed.weights[0] is the embedding matrix
        hs = self.transformer(projected_features, padding_mask, self.query_embed.weights[0], pos_embed)

        outputs_class = self.class_embed(hs)
        outputs_coord = keras.ops.sigmoid(self.bbox_embed(hs))

        last_output = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
        }

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
            "backbone": keras.saving.serialize(self.backbone),
            "transformer": keras.saving.serialize(self.transformer),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DETR":
        backbone_config = config.pop("backbone")
        transformer_config = config.pop("transformer")
        backbone = keras.saving.deserialize(backbone_config)
        transformer = keras.saving.deserialize(transformer_config)
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
    aux_loss: bool = True
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
        dropout=dropout
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