import math
import keras
from keras import ops, layers, initializers, activations
from typing import Optional, Union, Tuple, Dict, Any, Callable, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.transformer import TransformerLayer
from dl_techniques.layers.embedding.positional_embedding import PositionalEmbedding


# ---------------------------------------------------------------------
# Image Encoder Components
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ImageProjectionHead(keras.layers.Layer):
    """
    Image projection head with global pooling and dense projection.
    """

    def __init__(
            self,
            projection_dim: int,
            dropout_rate: float = 0.0,
            activation: Optional[Union[str, Callable]] = None,
            kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, initializers.Initializer] = 'zeros',
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if projection_dim <= 0:
            raise ValueError(f"projection_dim must be positive, got {projection_dim}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

        self.projection_dim = projection_dim
        self.dropout_rate = dropout_rate
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.global_pool = layers.GlobalAveragePooling2D(name='global_pool')
        self.dropout = layers.Dropout(self.dropout_rate, name='dropout')
        self.projection = layers.Dense(
            self.projection_dim,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            activation=None,
            name='projection'
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        pooled_shape = self.global_pool.compute_output_shape(input_shape)
        self.projection.build(pooled_shape)
        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        x = self.global_pool(inputs)
        x = self.dropout(x, training=training)
        x = self.projection(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], int]:
        return input_shape[0], self.projection_dim

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'projection_dim': self.projection_dim, 'dropout_rate': self.dropout_rate,
            'activation': activations.serialize(self.activation),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
        })
        return config


@keras.saving.register_keras_serializable()
class MobileClipImageEncoder(keras.Model):
    """
    MobileClip Image Encoder combining a backbone and a projection head.
    """

    def __init__(
            self,
            backbone_name: str = 'MobileNetV2',
            image_size: int = 224,
            projection_dim: int = 512,
            backbone_weights: Optional[str] = 'imagenet',
            backbone_trainable: bool = True,
            projection_dropout: float = 0.0,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.backbone_name = backbone_name
        self.image_size = image_size
        self.projection_dim = projection_dim
        self.backbone_weights = backbone_weights
        self.backbone_trainable = backbone_trainable
        self.projection_dropout = projection_dropout

        self.backbone = self._create_backbone()
        self.projection_head = ImageProjectionHead(
            projection_dim=self.projection_dim,
            dropout_rate=self.projection_dropout,
            name='projection_head'
        )

    def _create_backbone(self) -> keras.Model:
        try:
            backbone_class = getattr(keras.applications, self.backbone_name)
            backbone = backbone_class(
                include_top=False, weights=self.backbone_weights, pooling=None,
                input_shape=(self.image_size, self.image_size, 3)
            )
            backbone.trainable = self.backbone_trainable
            return backbone
        except AttributeError:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}.")

    def build(self, input_shape: Union[Tuple[int, ...], List[int]]) -> None:
        """Explicitly build sub-layers to ensure deterministic serialization."""
        self.backbone.build(input_shape)
        backbone_output_shape = self.backbone.compute_output_shape(input_shape)
        self.projection_head.build(backbone_output_shape)
        self.built = True

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        features = self.backbone(inputs, training=training)
        return self.projection_head(features, training=training)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'backbone_name': self.backbone_name, 'image_size': self.image_size,
            'projection_dim': self.projection_dim, 'backbone_weights': self.backbone_weights,
            'backbone_trainable': self.backbone_trainable,
            'projection_dropout': self.projection_dropout,
        })
        return config


# ---------------------------------------------------------------------
# Text Encoder Components
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MobileClipTextEncoder(keras.layers.Layer):
    """
    MobileClip Text Encoder using dl_techniques Transformer layers.
    """

    def __init__(
            self,
            vocab_size: int, max_seq_len: int, embed_dim: int, num_layers: int,
            num_heads: int, intermediate_size: int, projection_dim: int,
            dropout_rate: float = 0.0, attention_dropout_rate: float = 0.0,
            use_causal_mask: bool = True, embed_scale: Optional[float] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if vocab_size <= 0: raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if max_seq_len <= 0: raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if embed_dim <= 0: raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if num_layers <= 0: raise ValueError(f"num_layers must be positive, got {num_layers}")
        if num_heads <= 0: raise ValueError(f"num_heads must be positive, got {num_heads}")
        if intermediate_size <= 0: raise ValueError(f"intermediate_size must be positive, got {intermediate_size}")
        if projection_dim <= 0: raise ValueError(f"projection_dim must be positive, got {projection_dim}")

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.use_causal_mask = use_causal_mask
        self.embed_scale = embed_scale if embed_scale is not None else (embed_dim ** -0.5)

        self.token_embedding = layers.Embedding(self.vocab_size, self.embed_dim, name='token_embedding')
        self.positional_embedding = PositionalEmbedding(self.max_seq_len, self.embed_dim, self.dropout_rate, name='positional_embedding')
        self.transformer_layers = [
            TransformerLayer(
                hidden_size=self.embed_dim, num_heads=self.num_heads,
                intermediate_size=self.intermediate_size, dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                normalization_position='pre', name=f'transformer_layer_{i}'
            ) for i in range(self.num_layers)
        ]
        self.layer_norm = layers.LayerNormalization(name='final_layer_norm')
        self.projection_weights = None

    def build(self, input_shape: Union[Tuple[Optional[int], ...], List[Optional[int]]]) -> None:
        input_shape = tuple(input_shape)
        pos_input_shape = input_shape + (self.embed_dim,)
        self.token_embedding.build(input_shape)
        self.positional_embedding.build(pos_input_shape)
        for layer in self.transformer_layers:
            layer.build(pos_input_shape)
        self.layer_norm.build(pos_input_shape)
        self.projection_weights = self.add_weight(
            name='projection_weights', shape=(self.embed_dim, self.projection_dim),
            initializer='glorot_uniform', trainable=True,
        )
        super().build(input_shape)

    def call(self, text_tokens: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        token_emb = self.token_embedding(text_tokens) * self.embed_scale
        x = self.positional_embedding(token_emb, training=training)

        causal_mask = None
        if self.use_causal_mask:
            seq_len = ops.shape(x)[1]
            causal_mask = ops.triu(ops.ones((seq_len, seq_len)), k=1)

        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, attention_mask=causal_mask, training=training)

        x = self.layer_norm(x)
        eot_positions = ops.argmax(text_tokens, axis=-1)
        eot_mask = ops.one_hot(eot_positions, num_classes=ops.shape(x)[1], dtype=x.dtype)
        eot_embeddings = ops.sum(x * ops.expand_dims(eot_mask, axis=-1), axis=1)
        return ops.matmul(eot_embeddings, self.projection_weights)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], int]:
        return input_shape[0], self.projection_dim

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size, 'max_seq_len': self.max_seq_len,
            'embed_dim': self.embed_dim, 'num_layers': self.num_layers,
            'num_heads': self.num_heads, 'intermediate_size': self.intermediate_size,
            'projection_dim': self.projection_dim, 'dropout_rate': self.dropout_rate,
            'attention_dropout_rate': self.attention_dropout_rate,
            'use_causal_mask': self.use_causal_mask, 'embed_scale': self.embed_scale,
        })
        return config