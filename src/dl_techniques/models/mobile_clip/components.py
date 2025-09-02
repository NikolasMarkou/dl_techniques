import math
import keras
from keras import ops, layers, initializers, activations
from typing import Optional, Union, Tuple, Dict, Any, Callable

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

    This layer performs global average pooling on input feature maps followed
    by a linear projection to the target embedding dimension.

    **Architecture**:
    ```
    Input(shape=[batch, height, width, channels])
           ↓
    GlobalAveragePooling2D()
           ↓
    Dense(projection_dim)
           ↓
    Output(shape=[batch, projection_dim])
    ```

    Args:
        projection_dim: Integer, the output embedding dimension. Must be positive.
        dropout_rate: Float between 0 and 1, dropout rate applied after pooling.
            Defaults to 0.0.
        activation: Optional activation function applied after projection.
            Defaults to None.
        kernel_initializer: Initializer for the projection layer weights.
            Defaults to 'glorot_uniform'.
        bias_initializer: Initializer for the projection layer bias.
            Defaults to 'zeros'.
        **kwargs: Additional arguments for the Layer base class.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`.

    Output shape:
        2D tensor with shape: `(batch_size, projection_dim)`.

    Example:
        ```python
        # Basic usage
        projection_head = ImageProjectionHead(projection_dim=512)
        features = keras.random.normal((32, 7, 7, 2048))
        embeddings = projection_head(features)  # Shape: (32, 512)

        # With dropout and activation
        projection_head = ImageProjectionHead(
            projection_dim=256,
            dropout_rate=0.1,
            activation='gelu'
        )
        ```
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
            raise ValueError(f"projection_dim must be positive, got {projection_dim}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

        # Store configuration
        self.projection_dim = projection_dim
        self.dropout_rate = dropout_rate
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        # CREATE sub-layers in __init__ (OrchestrationLayer pattern)
        self.global_pool = layers.GlobalAveragePooling2D(name='global_pool')
        self.dropout = layers.Dropout(self.dropout_rate, name='dropout')
        self.projection = layers.Dense(
            self.projection_dim,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            activation=None,  # Activation is applied manually after
            name='projection'
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the dense projection layer."""
        super().build(input_shape)
        # The dense layer's input shape will be (batch_size, channels) after pooling.
        # The number of channels is the last dimension of the input to this layer.
        self.projection.build((None, input_shape[-1]))

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass computation."""
        x = self.global_pool(inputs)
        x = self.dropout(x, training=training)
        x = self.projection(x)

        if self.activation is not None:
            x = self.activation(x)

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        return (input_shape[0], self.projection_dim)

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

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MobileClipImageEncoder(keras.Model):
    """
    MobileClip Image Encoder combining backbone and projection head.

    This model wraps a backbone feature extractor with a custom projection head
    for embedding generation. It demonstrates the "Basic Custom Model" pattern
    from the guide, where Keras handles sub-layer building automatically.

    **Architecture**:
    ```
    Input Image
         ↓
    Backbone Feature Extractor
         ↓
    Feature Maps
         ↓
    ImageProjectionHead
         ↓
    Image Embedding
    ```

    Args:
        backbone_name: String, name of the backbone model. Currently supports
            models from keras.applications. Defaults to 'MobileNetV2'.
        projection_dim: Integer, the final output embedding dimension.
            Must be positive.
        backbone_weights: Optional string, pre-trained weights for backbone.
            Defaults to 'imagenet'.
        backbone_trainable: Boolean, whether backbone layers are trainable.
            Defaults to True.
        projection_dropout: Float, dropout rate in projection head.
            Defaults to 0.0.
        **kwargs: Additional arguments for the Model base class.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, 3)`.

    Output shape:
        2D tensor with shape: `(batch_size, projection_dim)`.

    Attributes:
        backbone: The feature extraction backbone model.
        projection_head: The projection head for embedding generation.

    Example:
        ```python
        # Basic usage
        encoder = MobileClipImageEncoder(
            backbone_name='MobileNetV2',
            projection_dim=512
        )

        images = keras.random.normal((32, 224, 224, 3))
        embeddings = encoder(images)  # Shape: (32, 512)

        # With custom configuration
        encoder = MobileClipImageEncoder(
            backbone_name='EfficientNetB0',
            projection_dim=256,
            backbone_trainable=False,
            projection_dropout=0.1
        )
        ```

    Note:
        The backbone model is created without the top classification layers
        (include_top=False) and uses global average pooling internally.
        The projection head then further processes these features.
    """

    def __init__(
            self,
            backbone_name: str = 'MobileNetV2',
            projection_dim: int = 512,
            backbone_weights: Optional[str] = 'imagenet',
            backbone_trainable: bool = True,
            projection_dropout: float = 0.0,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if projection_dim <= 0:
            raise ValueError(f"projection_dim must be positive, got {projection_dim}")
        if not (0.0 <= projection_dropout <= 1.0):
            raise ValueError(f"projection_dropout must be between 0 and 1, got {projection_dropout}")

        # Store configuration
        self.backbone_name = backbone_name
        self.projection_dim = projection_dim
        self.backbone_weights = backbone_weights
        self.backbone_trainable = backbone_trainable
        self.projection_dropout = projection_dropout

        # CREATE sub-models in __init__
        self.backbone = self._create_backbone()
        self.projection_head = ImageProjectionHead(
            projection_dim=self.projection_dim,
            dropout_rate=self.projection_dropout,
            name='projection_head'
        )

    def _create_backbone(self) -> keras.Model:
        """Create the backbone feature extractor."""
        try:
            # Get the backbone from keras.applications
            backbone_class = getattr(keras.applications, self.backbone_name)
            backbone = backbone_class(
                include_top=False,
                weights=self.backbone_weights,
                pooling=None,  # We handle pooling in the projection head
                input_shape=(224, 224, 3)
            )
            backbone.trainable = self.backbone_trainable
            return backbone

        except AttributeError:
            raise ValueError(
                f"Unsupported backbone: {self.backbone_name}. "
                f"Please use a model from keras.applications"
            )

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the model."""
        features = self.backbone(inputs, training=training)
        embeddings = self.projection_head(features, training=training)
        return embeddings

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        config = super().get_config()
        config.update({
            'backbone_name': self.backbone_name,
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
    MobileClip Text Encoder using dl_techniques TransformerLayer.

    This layer implements the text encoding pipeline using the existing
    TransformerLayer from dl_techniques framework. It demonstrates the
    "Composite Layer" pattern with both sub-layers and custom weights,
    following the guide's best practices.

    **Architecture**:
    ```
    Text Tokens
         ↓
    Token Embedding + Positional Embedding
         ↓
    Embedding Dropout
         ↓
    Stack of TransformerLayers
         ↓
    Layer Normalization
         ↓
    EOT Token Selection + Projection
         ↓
    Text Embedding
    ```

    Args:
        vocab_size: Integer, size of the vocabulary. Must be positive.
        max_seq_len: Integer, maximum sequence length. Must be positive.
        embed_dim: Integer, embedding dimension. Must be positive.
        num_layers: Integer, number of transformer layers. Must be positive.
        num_heads: Integer, number of attention heads per layer. Must be positive.
        intermediate_size: Integer, size of the feed-forward layer.
        projection_dim: Integer, final output embedding dimension.
        dropout_rate: Float, dropout rate for embeddings. Defaults to 0.0.
        attention_dropout_rate: Float, attention dropout rate. Defaults to 0.0.
        use_causal_mask: Boolean, whether to use causal masking. Defaults to True.
        embed_scale: Optional float, scaling factor for embeddings.
            If None, uses sqrt(embed_dim). Defaults to None.
        **kwargs: Additional arguments for the Layer base class.

    Input shape:
        2D tensor with shape: `(batch_size, sequence_length)`.

    Output shape:
        2D tensor with shape: `(batch_size, projection_dim)`.

    Attributes:
        token_embedding: Token embedding layer.
        positional_embedding: Positional embedding layer.
        embedding_dropout: Dropout layer for embeddings.
        transformer_layers: List of TransformerLayer instances.
        layer_norm: Final layer normalization.
        projection_weights: Projection weights for output embedding.

    Example:
        ```python
        # Basic usage
        text_encoder = MobileClipTextEncoder(
            vocab_size=50000,
            max_seq_len=512,
            embed_dim=512,
            num_layers=6,
            num_heads=8,
            intermediate_size=2048,
            projection_dim=512
        )

        text_tokens = keras.random.randint(0, 50000, (32, 128))
        embeddings = text_encoder(text_tokens)  # Shape: (32, 512)
        ```

    Note:
        This implementation uses the existing TransformerLayer from dl_techniques
        instead of custom transformer blocks, promoting code reuse and consistency.
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
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        if intermediate_size <= 0:
            raise ValueError(f"intermediate_size must be positive, got {intermediate_size}")
        if projection_dim <= 0:
            raise ValueError(f"projection_dim must be positive, got {projection_dim}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")
        if not (0.0 <= attention_dropout_rate <= 1.0):
            raise ValueError(f"attention_dropout_rate must be between 0 and 1, got {attention_dropout_rate}")

        # Store configuration
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

        # CREATE sub-layers in __init__
        self.token_embedding = layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embed_dim,
            name='token_embedding'
        )

        self.positional_embedding = PositionalEmbedding(
            max_seq_len=self.max_seq_len,
            dim=self.embed_dim,
            dropout=self.dropout_rate,
            name='positional_embedding'
        )

        # Create transformer layers using dl_techniques TransformerLayer
        self.transformer_layers = []
        for i in range(self.num_layers):
            self.transformer_layers.append(
                TransformerLayer(
                    hidden_size=self.embed_dim,
                    num_heads=self.num_heads,
                    intermediate_size=self.intermediate_size,
                    dropout_rate=self.dropout_rate,
                    attention_dropout_rate=self.attention_dropout_rate,
                    normalization_position='pre',  # Pre-LN is more stable
                    name=f'transformer_layer_{i}'
                )
            )

        self.layer_norm = layers.LayerNormalization(name='final_layer_norm')

        # Projection weights created in build()
        self.projection_weights = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer and create projection weights."""
        # Ensure input_shape is a tuple for robust serialization
        input_shape = tuple(input_shape)

        # Build sub-layers explicitly for robust serialization
        # Token embedding
        self.token_embedding.build(input_shape)

        # Positional embedding expects (batch, seq_len, embed_dim)
        pos_input_shape = input_shape + (self.embed_dim,)
        self.positional_embedding.build(pos_input_shape)

        # Transformer layers
        transformer_input_shape = pos_input_shape
        for layer in self.transformer_layers:
            layer.build(transformer_input_shape)

        # Layer norm
        self.layer_norm.build(transformer_input_shape)

        # Create projection weights
        self.projection_weights = self.add_weight(
            name='projection_weights',
            shape=(self.embed_dim, self.projection_dim),
            initializer='glorot_uniform',
            trainable=True,
        )

        super().build(input_shape)

    def call(
            self,
            text_tokens: keras.KerasTensor,
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass for text encoding."""
        # 1. Token embeddings with scaling
        token_emb = self.token_embedding(text_tokens) * self.embed_scale

        # 2. Add positional embeddings
        x = self.positional_embedding(token_emb, training=training)

        # 3. Create causal mask if needed
        if self.use_causal_mask:
            seq_len = ops.shape(x)[1]
            causal_mask = ops.triu(ops.ones((seq_len, seq_len)), k=1)
            causal_mask = ops.cast(causal_mask, dtype='bool')
            # Expand for batch dimension and num_heads
            batch_size = ops.shape(x)[0]
            causal_mask = ops.expand_dims(causal_mask, axis=0)  # (1, seq_len, seq_len)
            causal_mask = ops.tile(causal_mask, (batch_size, 1, 1))  # (batch, seq_len, seq_len)
        else:
            causal_mask = None

        # 4. Pass through transformer layers
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, attention_mask=causal_mask, training=training)

        # 5. Final layer normalization
        x = self.layer_norm(x, training=training)

        # 6. Extract EOT token embeddings
        # Find the position of the highest token ID in each sequence (EOT token)
        eot_positions = ops.argmax(text_tokens, axis=-1)  # Shape: (batch_size,)

        # Use one-hot masking to select the EOT embedding for each sequence.
        # This is a robust and backend-agnostic way to perform batched indexing.
        seq_len = ops.shape(x)[1]
        eot_mask = ops.one_hot(eot_positions, num_classes=seq_len, dtype=x.dtype)
        eot_mask = ops.expand_dims(eot_mask, axis=-1)  # Shape: (batch, seq_len, 1)

        # Multiply the sequence embeddings by the mask and sum along the sequence axis.
        # This zeros out all non-EOT embeddings, leaving only the EOT one.
        eot_embeddings = ops.sum(x * eot_mask, axis=1) # Shape: (batch, embed_dim)

        # 7. Project to final embedding space
        output_embeddings = ops.matmul(eot_embeddings, self.projection_weights)

        return output_embeddings

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        return (input_shape[0], self.projection_dim)

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

# ---------------------------------------------------------------------