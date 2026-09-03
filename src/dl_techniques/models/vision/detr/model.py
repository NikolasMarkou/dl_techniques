"""
DETR object detection: a CNN backbone feeding a transformer encoder-decoder
that turns a fixed set of learned object queries into box and class
predictions.

Earlier detectors produce tens of thousands of candidate boxes (anchors,
region proposals, grid cells) and prune them with a non-differentiable
non-maximum-suppression step. DETR instead treats detection as set
prediction. `num_queries` learned embeddings, shared across every image and
carrying no image content, pass through decoder self-attention so a query
that has claimed an object can signal that to the others, replacing NMS with
a learned interaction. Training matches this fixed-size prediction set to
the ground-truth set with a Hungarian bipartite matching (implemented in the
training loop, not here) and supervises unmatched predictions toward a
"no object" class, which is why the classifier emits `num_classes + 1`
outputs. `num_queries` is a hard ceiling on detections per image.

The forward path is backbone, 1x1 projection to `hidden_dim`, flatten to a
sequence, encoder, decoder, then two heads: a `Dense` for class logits and a
three-layer MLP for boxes passed through `sigmoid`, giving normalized
`cxcywh` coordinates in `[0, 1]`. `aux_loss=True`, the default, applies both
heads to every decoder layer and returns the intermediate predictions under
`aux_outputs`, which speeds convergence during training; inference needs
only `pred_logits` and `pred_boxes`. The backbone downloads ImageNet weights
through `keras.applications` and is frozen by default
(`backbone_trainable=False`); everything above it is not pretrained. See
:class:`DETR`'s docstring for where this implementation departs from the
paper.

References:
    - Carion et al., 2020. End-to-End Object Detection with Transformers.
      (https://arxiv.org/abs/2005.12872)
    - Vaswani et al., 2017. Attention Is All You Need.
      (https://arxiv.org/abs/1706.03762)
    - Kuhn, 1955. The Hungarian Method for the Assignment Problem.
      Naval Research Logistics Quarterly 2(1-2).
      The bipartite matching the training loop must supply.
    - He et al., 2015. Deep Residual Learning for Image Recognition.
      (https://arxiv.org/abs/1512.03385)
      The ResNet-50 backbone.
    - Zhu et al., 2021. Deformable DETR: Deformable Transformers for End-to-End
      Object Detection. (https://arxiv.org/abs/2010.04159)
      Diagnoses DETR's slow convergence and its weakness on small objects.
"""

import keras
from keras import layers, models
from typing import Optional, Dict, Any, List, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.transformers import TransformerLayer, TransformerDecoderLayer, FFNType, NormalizationType
from dl_techniques.layers.embedding.positional_embedding_sine_2d import PositionEmbeddingSine2D
from dl_techniques.utils.activation_serialization import (
    serialize_activation,
    deserialize_activation,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------
# Transformer Components
# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.detr.model")
class DetrTransformer(layers.Layer):
    """Encoder-decoder transformer that turns image features into per-query outputs.

    Architecture:

    .. code-block:: text

        src [B, H*W, D]   pos_embed [B, H*W, D]
              │                  │
              └────────┬─────────┘
                       ▼
        ┌─────────────────────────────┐
        │ Encoder stack (N layers)    │  self-attention, masked by `mask`
        └──────────────┬──────────────┘
                       ▼
                    memory [B, H*W, D]
                       │
        query_embed [Q, D] ──────────┐
                       │             ▼
        ┌─────────────────────────────┐
        │ Decoder stack (N layers)    │  self-attn over queries,
        │                             │  cross-attn over memory
        └──────────────┬──────────────┘
                       ▼
        per-layer outputs [B, Q, D], one per decoder layer

    :param hidden_dim: Transformer hidden dimension.
    :type hidden_dim: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param num_encoder_layers: Number of encoder layers.
    :type num_encoder_layers: int
    :param num_decoder_layers: Number of decoder layers.
    :type num_decoder_layers: int
    :param ffn_dim: Hidden dimension of the feed-forward network.
    :type ffn_dim: int
    :param dropout_rate: Dropout rate.
    :type dropout_rate: float
    :param activation: Activation function for the feed-forward network.
    :type activation: str
    :param normalization_type: Normalization type used throughout.
    :type normalization_type: str
    :param ffn_type: Feed-forward network variant.
    :type ffn_type: str
    :param kwargs: Additional layer arguments.

    Input shape:
        Tuple of:
        - src: ``(batch_size, H*W, hidden_dim)``, flattened image features.
        - mask: ``(batch_size, H*W)``, padding mask.
        - query_embed: ``(num_queries, hidden_dim)``, object query embeddings.
        - pos_embed: ``(batch_size, H*W, hidden_dim)``, positional encodings.

    Output shape:
        List of ``(batch_size, num_queries, hidden_dim)`` tensors, one per
        decoder layer.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        ffn_dim: int = 2048,
        dropout_rate: float = 0.1,
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
        self.dropout_rate = dropout_rate
        self.activation = deserialize_activation(activation)
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
                    dropout_rate=dropout_rate,
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
                    dropout_rate=dropout_rate,
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
        """Run the encoder stack, then the decoder stack.

        :param src: Source features, shape ``(batch_size, H*W, hidden_dim)``.
        :type src: keras.KerasTensor
        :param mask: Key keep mask, shape ``(batch_size, H*W)``, 1 for a real
            feature position and 0 for one that came from image padding.
            ``None`` attends to everything. This is the inverse of the
            ``padding_mask`` :class:`DETR` accepts, which is true for
            padding; :meth:`DETR.call` does the inversion.
        :type mask: Optional[keras.KerasTensor]
        :param query_embed: Object query embeddings, shape ``(num_queries, hidden_dim)``.
        :type query_embed: keras.KerasTensor
        :param pos_embed: Positional encodings, shape ``(batch_size, H*W, hidden_dim)``.
        :type pos_embed: keras.KerasTensor
        :param training: Whether the layer runs in training or inference mode.
        :type training: Optional[bool]
        :return: List of decoder outputs, one per layer.
        :rtype: List[keras.KerasTensor]
        """
        # DECISION plan-2026-08-14T233721-d4f9beb2/D-046: pass the 2-D (B, S) key mask straight through, never materialize (B, T, S).
        # `MultiHeadCrossAttention` broadcasts it to (B, 1, 1, S) itself; the square form costs O(S^2) for nothing. See decisions.md.

        # Encoder forward pass
        memory = src
        for encoder_layer in self.encoder_layers:
            # Add positional encoding to the input for each encoder layer
            memory = encoder_layer(
                memory + pos_embed, attention_mask=mask, training=training
            )

        # Decoder forward pass (fix 1f)
        batch_size = keras.ops.shape(src)[0]
        num_queries = keras.ops.shape(query_embed)[0]
        query_embed_expanded = keras.ops.tile(
            keras.ops.expand_dims(query_embed, axis=0),
            [batch_size, 1, 1]
        )
        # Materialise the decoder's zero query slot in the layer's COMPUTE dtype.
        # `keras.ops.zeros` with no `dtype=` returns float32 regardless of the
        # active mixed-precision policy, and the resulting float32 tensor meets a
        # float16 `query_embed_expanded` two statements below, so the raise landed
        # on the addition rather than here.
        tgt = keras.ops.zeros(
            (batch_size, num_queries, self.hidden_dim), dtype=self.compute_dtype
        )

        decoder_outputs = []
        for decoder_layer in self.decoder_layers:
            tgt = decoder_layer(
                tgt + query_embed_expanded,
                memory,
                cross_attention_mask=mask,
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
            "dropout_rate": self.dropout_rate,
            "activation": serialize_activation(self.activation),
            "normalization_type": self.normalization_type,
            "ffn_type": self.ffn_type,
        })
        return config


# ---------------------------------------------------------------------
# Main DETR Model
# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.detr.model")
class DETR(models.Model):
    """
    End-to-end object detection: backbone, transformer, two prediction heads.

    Architecture:

    .. code-block:: text

        images [B, H, W, 3]        padding_mask [B, H, W]
              │                          │
              ▼                          │
        ┌──────────────┐                 │
        │ backbone(CNN)│                 │
        │ (pretrained, │                 │
        │  frozen)     │                 │
        └──────┬───────┘                 │
               ▼                         ▼ (nearest-downsample to feature grid)
        ┌──────────────┐         ┌───────────────┐
        │ 1x1 Conv     │         │ keep mask [B,S]│
        │ → hidden_dim │         └───────┬────────┘
        └──────┬───────┘                 │
               ├─────────────────────────┤
               ▼                         ▼
        ┌────────────────────────────────────┐
        │ pos_embed: PositionEmbeddingSine2D  │  (computed on the projected
        │ (on the projected feature map)      │   feature map, not the image)
        └──────────────────┬───────────────────┘
                           ▼
        ┌────────────────────────────────────┐
        │ DetrTransformer                     │
        └──────────────────┬───────────────────┘
                           ▼
              per-decoder-layer outputs
                    │              │
                    ▼              ▼
             class_embed      bbox_embed (3-layer MLP, sigmoid)
                    │              │
                    ▼              ▼
             pred_logits      pred_boxes
        [B, Q, num_classes+1] [B, Q, 4] (cxcywh, [0, 1])

        aux_loss=True also applies both heads to every non-final decoder
        layer's output and returns them under aux_outputs.

    This implementation departs from the paper in a few places. The padding
    mask is honoured only above the backbone: it is downsampled to the
    stride-16 feature grid and passed to encoder self-attention and decoder
    cross-attention, but the backbone convolutions themselves still see the
    padded canvas, matching the reference implementation. Positional
    encodings are added to the running encoder `memory` at every layer
    (accumulating across the stack) rather than only to queries and keys, and
    the decoder does the same with the query embeddings. The backbone is
    tapped at `conv4_block6_out` (C4, stride 16) rather than C5, and there is
    no final decoder layer normalization, so auxiliary outputs are read raw
    from each decoder layer.

    :param num_classes: Number of object classes, excluding "no object".
    :type num_classes: int
    :param num_queries: Maximum number of detections per image.
    :type num_queries: int
    :param backbone: A Keras CNN model used for feature extraction.
    :type backbone: keras.Model
    :param transformer: The DETR transformer module.
    :type transformer: DetrTransformer
    :param hidden_dim: Transformer dimensionality. Must be a multiple of 4:
        the sine position encoding takes `hidden_dim // 2` features per axis,
        and that value must itself be even.
    :type hidden_dim: int
    :param aux_loss: Whether to return predictions from every intermediate
        decoder layer for auxiliary loss calculation.
    :type aux_loss: bool
    :param kwargs: Additional model arguments.

    Input shape:
        Tuple of:
        - images: ``(batch_size, height, width, 3)``.
        - padding_mask: ``(batch_size, height, width)`` boolean mask.

    Output shape:
        Dictionary containing:
        - pred_logits: ``(batch_size, num_queries, num_classes + 1)``.
        - pred_boxes: ``(batch_size, num_queries, 4)``.
        - aux_outputs: list of dicts, if ``aux_loss=True``.
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
        # DECISION plan-2026-08-28T181715-3870472c/D-007: keep the `% 4` check; do not weaken it to `% 2`.
        # `hidden_dim // 2` becomes `PositionEmbeddingSine2D`'s `num_pos_feats`, which must itself be even. See decisions.md.
        if hidden_dim % 4 != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be a multiple of 4: the sine "
                f"position encoding receives num_pos_feats = hidden_dim // 2 = "
                f"{hidden_dim // 2}, and that value must ITSELF be even "
                f"because PositionEmbeddingSine2D splits it between its sine "
                f"and cosine halves. Use "
                f"hidden_dim = {((hidden_dim + 3) // 4) * 4}."
            )

        self.num_classes = num_classes
        self.num_queries = num_queries
        self.backbone = backbone
        self.transformer = transformer
        self.hidden_dim = hidden_dim
        self.aux_loss = aux_loss

        # Prediction heads
        self.class_embed = layers.Dense(num_classes + 1, name="class_embed")

        # Box prediction head: the paper's 3-layer perceptron,
        # `Dense(d) -> ReLU -> Dense(d) -> ReLU -> Dense(4)`.
        # DECISION plan-2026-08-14T233721-d4f9beb2/D-047: build this Sequential directly, not via `create_ffn_layer('mlp', ...)`.
        # That factory key builds a 2-layer MLPBlock, not the paper's 3-layer one, and no depth-configurable MLP exists in layers/ffn/. See decisions.md.
        self.bbox_embed = keras.Sequential(
            [
                layers.Dense(hidden_dim, activation='relu', name="bbox_fc1"),
                layers.Dense(hidden_dim, activation='relu', name="bbox_fc2"),
                layers.Dense(4, name="bbox_fc3"),
            ],
            name="bbox_embed",
        )

        # Query embeddings
        self.query_embed = layers.Embedding(num_queries, hidden_dim, name="query_embed")

        # Input projection
        self.input_proj = layers.Conv2D(hidden_dim, kernel_size=1, name="input_proj")

        # Positional embedding
        self.pos_embed = PositionEmbeddingSine2D(num_pos_feats=hidden_dim // 2, name="pos_embed")

    def build(self, input_shape) -> None:
        """Build DETR and every sublayer so a ``.keras`` round trip can restore weights.

        Keras calls ``load_own_variables`` on each sublayer during restore;
        an unbuilt sublayer has no variables, so the restore raises when the
        saved store holds more entries than that empty list. Building here
        gives every sublayer its variables before weights are loaded.

        :param input_shape: Shape of the input tensor(s).
        :type input_shape: Any
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
        """Run the backbone, transformer, and prediction heads.

        :param inputs: Tuple of ``(images, padding_mask)``.
        :type inputs: Tuple[keras.KerasTensor, keras.KerasTensor]
        :param training: Whether the model runs in training or inference mode.
        :type training: Optional[bool]
        :return: Dictionary with ``pred_logits``, ``pred_boxes``, and,
            when ``aux_loss=True``, ``aux_outputs``.
        :rtype: Dict[str, Any]
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

        # DECISION plan-2026-08-14T233721-d4f9beb2/D-046: downsample the padding mask with nearest interpolation, never area/bilinear.
        # The attention keep predicate is binary (`> 0`); an interpolated 0.5 at a boundary cell would read as full keep. See decisions.md.
        key_keep_mask = None
        if padding_mask is not None:
            mask_f = keras.ops.cast(padding_mask, projected_features.dtype)
            if len(mask_f.shape) == 3:
                mask_f = keras.ops.expand_dims(mask_f, axis=-1)
            mask_small = keras.ops.image.resize(
                mask_f, size=(height, width), interpolation="nearest",
            )
            key_keep_mask = 1.0 - keras.ops.reshape(
                mask_small, (batch_size, height * width)
            )

        hs = self.transformer(
            src, key_keep_mask, query_embed_weights, pos_embed_flat,
            training=training,
        )

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
        """Deserialize a model from configuration, migrating a pre-D-007 archive.

        An archive written before the multiple-of-4 `hidden_dim` rule can
        carry a non-conforming value. That value is rounded up here with a
        warning rather than rejected, and the serialized transformer
        sub-config is updated to match so the two widths stay consistent.
        Such an archive's position encoder could never complete a forward
        pass, so rounding up cannot break anything that previously worked.

        :param config: Configuration dictionary from :meth:`get_config`.
        :type config: Dict[str, Any]
        :return: DETR model instance.
        :rtype: DETR
        """
        config = dict(config)
        hidden_dim = config.get("hidden_dim")
        if isinstance(hidden_dim, int) and hidden_dim > 0 and hidden_dim % 4 != 0:
            substitute = ((hidden_dim + 3) // 4) * 4
            logger.warning(
                "DETR config carries hidden_dim=%d, whose sine width "
                "num_pos_feats=%d is odd; this archive predates the "
                "multiple-of-4 requirement and its position encoder could "
                "never run a forward pass. Substituting hidden_dim=%d "
                "(num_pos_feats=%d). Every hidden_dim-wide weight changes "
                "width from %d to %d, so stored weights will not match.",
                hidden_dim, hidden_dim // 2, substitute, substitute // 2,
                hidden_dim, substitute,
            )
            config["hidden_dim"] = substitute
            transformer_config = config.get("transformer")
            if (isinstance(transformer_config, dict)
                    and isinstance(transformer_config.get("config"), dict)
                    and transformer_config["config"].get("hidden_dim")
                    == hidden_dim):
                transformer_config = dict(transformer_config)
                transformer_config["config"] = dict(
                    transformer_config["config"])
                transformer_config["config"]["hidden_dim"] = substitute
                config["transformer"] = transformer_config
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
    dropout_rate: float = 0.1,
    aux_loss: bool = True,
    activation: str = "relu",
    normalization_type: str = "layer_norm",
    ffn_type: str = "mlp"
) -> DETR:
    """Build a DETR model from a named backbone and transformer hyperparameters.

    :param num_classes: Number of object detection classes.
    :type num_classes: int
    :param num_queries: Number of object queries.
    :type num_queries: int
    :param backbone_name: Name of the CNN backbone; only ``"resnet50"`` is implemented.
    :type backbone_name: str
    :param backbone_trainable: Whether the backbone weights are fine-tuned.
    :type backbone_trainable: bool
    :param hidden_dim: Transformer dimensionality. Must be a multiple of 4.
    :type hidden_dim: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param num_encoder_layers: Number of encoder layers.
    :type num_encoder_layers: int
    :param num_decoder_layers: Number of decoder layers.
    :type num_decoder_layers: int
    :param ffn_dim: Hidden dimension of the transformer's feed-forward networks.
    :type ffn_dim: int
    :param dropout_rate: Dropout rate used in the transformer.
    :type dropout_rate: float
    :param aux_loss: Whether the model outputs predictions from intermediate layers.
    :type aux_loss: bool
    :param activation: Activation function for the feed-forward network.
    :type activation: str
    :param normalization_type: Normalization type used throughout.
    :type normalization_type: str
    :param ffn_type: Feed-forward network variant.
    :type ffn_type: str
    :return: A DETR Keras model instance.
    :rtype: DETR
    :raises NotImplementedError: If `backbone_name` is not ``"resnet50"``.
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
        dropout_rate=dropout_rate,
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

# ---------------------------------------------------------------------
