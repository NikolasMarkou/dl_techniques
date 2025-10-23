import keras
from keras import ops, layers, initializers, activations
from typing import Optional, Union, Tuple, Dict, Any, Callable, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.transformers import TransformerLayer
from dl_techniques.layers.embedding.positional_embedding import PositionalEmbedding


# ---------------------------------------------------------------------
# Image Encoder Components
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ImageProjectionHead(keras.layers.Layer):
    """
    Projects image feature maps into a fixed-size embedding.

    This layer takes 4D feature maps from a convolutional backbone, applies
    global average pooling to summarize spatial information, and then projects
    the result into a final embedding vector using a Dense layer. It includes
    optional dropout and activation for regularization and non-linearity.

    **Intent**: To serve as the final stage of an image encoder, creating a
    fixed-size representation of an image suitable for contrastive learning
    or other downstream tasks.

    **Architecture**:
    ```
    Input(shape=[B, H, W, C])
           ↓
    GlobalAveragePooling2D()
           ↓
    Output(shape=[B, C])
           ↓
    Dropout(rate=dropout_rate)
           ↓
    Dense(projection_dim)
           ↓
    Activation(·) (if provided)
           ↓
    Output(shape=[B, projection_dim])
    ```

    Args:
        projection_dim: Integer, dimensionality of the output embedding space.
            Must be positive.
        dropout_rate: Float between 0 and 1. Fraction of the pooled units to
            drop. Defaults to 0.0.
        activation: Optional activation function for the final output. Can be
            a string name or a callable. Defaults to None (linear activation).
        kernel_initializer: Initializer for the projection layer's kernel.
            Defaults to 'glorot_uniform'.
        bias_initializer: Initializer for the projection layer's bias.
            Defaults to 'zeros'.
        **kwargs: Additional arguments for the Layer base class.

    Input shape:
        4D tensor with shape `(batch_size, height, width, channels)`.

    Output shape:
        2D tensor with shape `(batch_size, projection_dim)`.

    Attributes:
        global_pool: GlobalAveragePooling2D layer.
        dropout: Dropout layer for regularization.
        projection: Dense layer for the final projection.
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

        # Validate inputs
        if projection_dim <= 0:
            raise ValueError(
                f"projection_dim must be positive, got {projection_dim}"
            )
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(
                f"dropout_rate must be between 0 and 1, got {dropout_rate}"
            )

        # Store ALL configuration
        self.projection_dim = projection_dim
        self.dropout_rate = dropout_rate
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        # CREATE all sub-layers in __init__ (they are unbuilt)
        self.global_pool = layers.GlobalAveragePooling2D(name='global_pool')
        self.dropout = layers.Dropout(self.dropout_rate, name='dropout')
        self.projection = layers.Dense(
            units=self.projection_dim,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            activation=None,  # Activation is applied manually in call
            name='projection'
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers.

        CRITICAL: Explicitly build each sub-layer for robust serialization.
        """
        # Build sub-layers in computational order
        self.global_pool.build(input_shape)

        # Compute intermediate shapes to build subsequent layers
        pooled_shape = self.global_pool.compute_output_shape(input_shape)
        self.dropout.build(pooled_shape)
        self.projection.build(pooled_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through sub-layers."""
        x = self.global_pool(inputs)
        x = self.dropout(x, training=training)
        x = self.projection(x)

        if self.activation is not None:
            x = self.activation(x)
        return x

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], int]:
        """Compute the output shape."""
        return input_shape[0], self.projection_dim

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'projection_dim': self.projection_dim,
            'dropout_rate': self.dropout_rate,
            'activation': activations.serialize(self.activation),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
        })
        return config


@keras.saving.register_keras_serializable()
class MobileClipImageEncoder(keras.Model):
    """
    MobileClip Image Encoder combining a backbone and a projection head.

    This model encapsulates the image processing pipeline for a CLIP-like
    model, using a pre-trained convolutional network (the backbone) to
    extract features, and an `ImageProjectionHead` to map these features
    into a final embedding space.

    **Intent**: To provide a complete, serializable image encoder that can be
    trained end-to-end, with options for using different backbones and
    controlling their trainability.

    **Architecture**:
    ```
    Input(shape=[B, image_size, image_size, 3])
           ↓
    Backbone (e.g., MobileNetV2)
           ↓
    Feature Maps(shape=[B, H, W, C])
           ↓
    ImageProjectionHead(projection_dim)
           ↓
    Output(shape=[B, projection_dim])
    ```

    Args:
        backbone_name: String name of a `keras.applications` model to use
            as the backbone. Defaults to 'MobileNetV2'.
        image_size: Integer, the height and width of the input images.
            Defaults to 224.
        projection_dim: Integer, dimensionality of the final output embedding.
            Defaults to 512.
        backbone_weights: String or None. Weights to initialize the backbone
            with (e.g., 'imagenet'). Defaults to 'imagenet'.
        backbone_trainable: Boolean, whether to fine-tune the backbone weights.
            Defaults to True.
        projection_dropout: Float between 0 and 1. Dropout rate for the
            projection head. Defaults to 0.0.
        **kwargs: Additional arguments for the Model base class.

    Input shape:
        4D tensor with shape `(batch_size, image_size, image_size, 3)`.

    Output shape:
        2D tensor with shape `(batch_size, projection_dim)`.

    Attributes:
        backbone: The instantiated Keras application model.
        projection_head: The `ImageProjectionHead` layer.
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

        # Store configuration
        self.backbone_name = backbone_name
        self.image_size = image_size
        self.projection_dim = projection_dim
        self.backbone_weights = backbone_weights
        self.backbone_trainable = backbone_trainable
        self.projection_dropout = projection_dropout

        # CREATE all sub-layers in __init__
        self.backbone = self._create_backbone()
        self.projection_head = ImageProjectionHead(
            projection_dim=self.projection_dim,
            dropout_rate=self.projection_dropout,
            name='projection_head'
        )

    def _create_backbone(self) -> keras.Model:
        """Instantiates the backbone model from keras.applications."""
        try:
            backbone_class = getattr(keras.applications, self.backbone_name)
            backbone = backbone_class(
                include_top=False,
                weights=self.backbone_weights,
                pooling=None,
                input_shape=(self.image_size, self.image_size, 3)
            )
            backbone.trainable = self.backbone_trainable
            return backbone
        except AttributeError:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}.")

    def build(self, input_shape: Union[Tuple[int, ...], List[int]]) -> None:
        """
        Explicitly build sub-layers to ensure deterministic serialization.

        While Keras Models often handle sub-layer building automatically, an
        explicit build method provides more robust and predictable behavior,
        especially for complex models.
        """
        self.backbone.build(input_shape)
        backbone_output_shape = self.backbone.compute_output_shape(input_shape)
        self.projection_head.build(backbone_output_shape)
        self.built = True

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the backbone and projection head."""
        features = self.backbone(inputs, training=training)
        return self.projection_head(features, training=training)

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration for serialization."""
        config = super().get_config()
        config.update({
            'backbone_name': self.backbone_name,
            'image_size': self.image_size,
            'projection_dim': self.projection_dim,
            'backbone_weights': self.backbone_weights,
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
    MobileClip Text Encoder using a stack of Transformer layers.

    This layer implements a full text encoding pipeline, including token
    and positional embeddings, a stack of Transformer blocks for contextual
    encoding, and a final projection. It is designed to be configurable and
    leverages framework components like `TransformerLayer`.

    **Intent**: To create a powerful and serializable text encoder that maps
    sequences of tokens to a fixed-size embedding, suitable for contrastive
    learning against image embeddings.

    **Architecture**:
    ```
    Input(shape=[B, T])            (Token IDs)
           ↓
    TokenEmbedding
           ↓
    PositionalEmbedding
           ↓
    N x TransformerLayer
           ↓
    LayerNormalization
           ↓
    Extract EOT Token Embedding
           ↓
    Projection (MatMul with projection_weights)
           ↓
    Output(shape=[B, projection_dim])
    ```

    Args:
        vocab_size: Integer, size of the token vocabulary.
        max_seq_len: Integer, maximum sequence length for positional embeddings.
        embed_dim: Integer, dimensionality of token and positional embeddings.
        num_layers: Integer, number of Transformer layers to stack.
        num_heads: Integer, number of attention heads in each Transformer layer.
        intermediate_size: Integer, size of the feed-forward network in
            each Transformer layer.
        projection_dim: Integer, dimensionality of the final output embedding.
        dropout_rate: Float, dropout rate for embeddings and FFNs.
        attention_dropout_rate: Float, dropout rate for attention weights.
        use_causal_mask: Boolean, whether to apply a causal mask in attention.
        embed_scale: Float, scaling factor for token embeddings. Defaults to
            `embed_dim ** -0.5`.
        **kwargs: Additional arguments for the Layer base class.

    Input shape:
        2D tensor of token IDs with shape `(batch_size, sequence_length)`.

    Output shape:
        2D tensor of embeddings with shape `(batch_size, projection_dim)`.
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
        projection_dim: int,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        use_causal_mask: bool = True,
        embed_scale: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if intermediate_size <= 0:
            raise ValueError(
                f"intermediate_size must be positive, got {intermediate_size}"
            )
        if projection_dim <= 0:
            raise ValueError(
                f"projection_dim must be positive, got {projection_dim}"
            )

        # Store ALL configuration
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

        # CREATE all sub-layers in __init__
        self.token_embedding = layers.Embedding(
            self.vocab_size, self.embed_dim, name='token_embedding'
        )
        self.positional_embedding = PositionalEmbedding(
            self.max_seq_len, self.embed_dim, self.dropout_rate,
            name='positional_embedding'
        )
        self.transformer_layers = [
            TransformerLayer(
                hidden_size=self.embed_dim,
                num_heads=self.num_heads,
                intermediate_size=self.intermediate_size,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                normalization_position='pre',
                name=f'transformer_layer_{i}'
            ) for i in range(self.num_layers)
        ]
        self.layer_norm = layers.LayerNormalization(name='final_layer_norm')

        # Initialize weight attributes - created in build()
        self.projection_weights = None

    def build(
        self,
        input_shape: Union[Tuple[Optional[int], ...], List[Optional[int]]]
    ) -> None:
        """Create weights and build all sub-layers."""
        input_shape = tuple(input_shape)
        pos_input_shape = input_shape + (self.embed_dim,)

        # Build sub-layers in computational order
        self.token_embedding.build(input_shape)
        self.positional_embedding.build(pos_input_shape)
        for layer in self.transformer_layers:
            layer.build(pos_input_shape)
        self.layer_norm.build(pos_input_shape)

        # Create layer's own weights
        self.projection_weights = self.add_weight(
            name='projection_weights',
            shape=(self.embed_dim, self.projection_dim),
            initializer='glorot_uniform',
            trainable=True,
        )

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        text_tokens: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass for text encoding."""
        token_emb = self.token_embedding(text_tokens) * self.embed_scale
        x = self.positional_embedding(token_emb, training=training)

        causal_mask = None
        if self.use_causal_mask:
            seq_len = ops.shape(x)[1]
            # Create a lower-triangular mask to prevent attending to future tokens.
            # The mask should be 3D to be broadcastable across the batch and head dimensions.
            causal_mask = ops.tril(ops.ones((1, seq_len, seq_len)))

        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, attention_mask=causal_mask, training=training)

        x = self.layer_norm(x)

        # Extract the features of the EOT (end-of-text) token.
        eot_positions = ops.argmax(text_tokens, axis=-1)
        eot_mask = ops.one_hot(
            eot_positions, num_classes=ops.shape(x)[1], dtype=x.dtype
        )
        # Use the mask to gather the EOT embeddings
        eot_embeddings = ops.sum(x * ops.expand_dims(eot_mask, axis=-1), axis=1)

        return ops.matmul(eot_embeddings, self.projection_weights)

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], int]:
        """Compute the output shape."""
        return input_shape[0], self.projection_dim

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'max_seq_len': self.max_seq_len,
            'embed_dim': self.embed_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'intermediate_size': self.intermediate_size,
            'projection_dim': self.projection_dim,
            'dropout_rate': self.dropout_rate,
            'attention_dropout_rate': self.attention_dropout_rate,
            'use_causal_mask': self.use_causal_mask,
            'embed_scale': self.embed_scale,
        })
        return config