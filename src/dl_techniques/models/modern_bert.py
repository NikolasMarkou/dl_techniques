import keras
from dataclasses import dataclass
from keras import ops, initializers
from typing import Optional, Union, Any, Dict, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..utils.logger import logger
from ..layers.transformer import TransformerLayer
from ..layers.rotary_position_embedding import RotaryPositionEmbedding


# ---------------------------------------------------------------------

@dataclass
class ModernBertConfig:
    vocab_size: int = 50368
    hidden_size: int = 768
    num_layers: int = 22
    num_heads: int = 12
    intermediate_size: int = 1152
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0
    classifier_dropout: Optional[float] = None
    normalization_type: str = "layer_norm"
    use_bias: bool = False
    rope_theta_local: float = 10000.0
    rope_theta_global: float = 160000.0
    rope_max_wavelength: int = 10000
    global_attention_interval: int = 3
    local_attention_window_size: int = 128

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format."""
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModernBertConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ModernBertEmbeddings(keras.layers.Layer):
    """
    ModernBERT embeddings layer without absolute position embeddings.
    ... (rest of the class is correct) ...
    """

    def __init__(self, config: ModernBertConfig, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config = config

        # Initialize sublayers to None - created in build()
        self.word_embeddings = None
        self.token_type_embeddings = None
        self.layer_norm = None
        self.dropout = None

        # Store build state
        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the embedding layers."""
        if self.built:
            return

        self._build_input_shape = input_shape

        # Create embeddings
        self.word_embeddings = keras.layers.Embedding(
            self.config.vocab_size,
            self.config.hidden_size,
            embeddings_initializer=initializers.TruncatedNormal(
                stddev=self.config.initializer_range
            ),
            name="word_embeddings"
        )

        self.token_type_embeddings = keras.layers.Embedding(
            self.config.type_vocab_size,
            self.config.hidden_size,
            embeddings_initializer=initializers.TruncatedNormal(
                stddev=self.config.initializer_range
            ),
            name="token_type_embeddings"
        )

        # Normalization and dropout
        self.layer_norm = keras.layers.LayerNormalization(
            epsilon=self.config.layer_norm_eps,
            center=self.config.use_bias,  # Corrected: use `center` for Keras 3
            name="layer_norm"
        )

        self.dropout = keras.layers.Dropout(
            self.config.hidden_dropout_prob,
            name="dropout"
        )

        # Build sublayers
        embedding_shape = tuple(input_shape) + (self.config.hidden_size,)
        self.word_embeddings.build(input_shape)
        self.token_type_embeddings.build(input_shape)
        self.layer_norm.build(embedding_shape)

        super().build(input_shape)

    def call(
            self,
            input_ids: keras.KerasTensor,
            token_type_ids: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through embeddings."""
        seq_length = ops.shape(input_ids)[1]

        if token_type_ids is None:
            token_type_ids = ops.zeros(
                (ops.shape(input_ids)[0], seq_length),
                dtype="int32"
            )

        word_embeds = self.word_embeddings(input_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)
        embeddings = word_embeds + token_type_embeds

        embeddings = self.layer_norm(embeddings, training=training)
        embeddings = self.dropout(embeddings, training=training)
        return embeddings

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({'config': self.config.to_dict()})
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration."""
        return {'input_shape': self._build_input_shape}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build from configuration."""
        if config.get('input_shape') is not None:
            self.build(config['input_shape'])

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ModernBertEmbeddings':
        """Create layer from configuration."""
        bert_config = ModernBertConfig.from_dict(config['config'])
        return cls(config=bert_config)


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ModernBertAttention(keras.layers.Layer):
    """
    ModernBERT attention layer with RoPE and sliding window support.
    ... (rest of the class is correct) ...
    """

    def __init__(
            self,
            config: ModernBertConfig,
            is_global: bool,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.config = config
        self.is_global = is_global
        self.mha = None
        self.rotary_embedding = None
        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        if self.built:
            return
        self._build_input_shape = input_shape
        self.mha = keras.layers.MultiHeadAttention(
            num_heads=self.config.num_heads,
            key_dim=self.config.hidden_size // self.config.num_heads,
            dropout=self.config.attention_probs_dropout_prob,
            use_bias=self.config.use_bias,
            kernel_initializer=initializers.TruncatedNormal(stddev=self.config.initializer_range),
            name="multi_head_attention"
        )
        self.rotary_embedding = RotaryPositionEmbedding(
            head_dim=self.config.hidden_size // self.config.num_heads,
            max_seq_len=8192,
            rope_theta=(self.config.rope_theta_global if self.is_global else self.config.rope_theta_local),
            name="rotary_embedding"
        )
        self.mha.build(query_shape=input_shape, value_shape=input_shape)
        # Rotary embedding build might depend on a 4D shape, build it lazily or with a dummy shape
        dummy_4d_shape = input_shape[:-1] + (self.config.num_heads, self.config.hidden_size // self.config.num_heads)
        self.rotary_embedding.build(dummy_4d_shape)
        super().build(input_shape)

    def _apply_rotary_pos_emb(self, tensor: keras.KerasTensor, cos_emb: keras.KerasTensor,
                              sin_emb: keras.KerasTensor) -> keras.KerasTensor:
        x1, x2 = ops.split(tensor, 2, axis=-1)
        half_rot_tensor = ops.concatenate([-x2, x1], axis=-1)
        return (tensor * cos_emb) + (half_rot_tensor * sin_emb)

    def _create_sliding_window_mask(self, seq_len: int) -> keras.KerasTensor:
        positions = ops.arange(seq_len, dtype="int32")
        mask = ops.abs(positions[:, None] - positions[None, :]) < self.config.local_attention_window_size
        return ops.cast(mask, "bool")

    def call(self, hidden_states: keras.KerasTensor, attention_mask: Optional[keras.KerasTensor] = None,
             training: Optional[bool] = None) -> keras.KerasTensor:
        seq_len = ops.shape(hidden_states)[1]
        batch_size = ops.shape(hidden_states)[0]

        # Reshape hidden states for multi-head processing FIRST
        hidden_states_reshaped = ops.reshape(
            hidden_states,
            (batch_size, seq_len, self.config.num_heads, -1)
        )

        # FIX: Get rotary embeddings based on the 4D reshaped tensor
        cos_emb, sin_emb = self.rotary_embedding(hidden_states_reshaped)

        # Apply RoPE to query and key
        query = self._apply_rotary_pos_emb(hidden_states_reshaped, cos_emb, sin_emb)
        key = self._apply_rotary_pos_emb(hidden_states_reshaped, cos_emb, sin_emb)

        # Reshape back
        query = ops.reshape(query, (batch_size, seq_len, self.config.hidden_size))
        key = ops.reshape(key, (batch_size, seq_len, self.config.hidden_size))
        value = hidden_states  # Value doesn't get RoPE

        # Create attention mask
        final_attention_mask = attention_mask
        if not self.is_global:
            sliding_mask = self._create_sliding_window_mask(seq_len)
            if attention_mask is not None:
                expanded_padding_mask = ops.expand_dims(attention_mask, axis=1)
                final_attention_mask = ops.logical_and(expanded_padding_mask, sliding_mask)
            else:
                final_attention_mask = sliding_mask

        return self.mha(
            query=query,
            value=value,
            key=key,
            attention_mask=final_attention_mask,
            training=training
        )

    # ... from_config, get_config, etc. remain the same ...
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'config': self.config.to_dict(),
            'is_global': self.is_global
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        return {'input_shape': self._build_input_shape}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        if config.get('input_shape') is not None:
            self.build(config['input_shape'])

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ModernBertAttention':
        bert_config = ModernBertConfig.from_dict(config['config'])
        return cls(config=bert_config, is_global=config['is_global'])


# ... ModernBertFFN and ModernBertTransformerLayer are correct ...
@keras.saving.register_keras_serializable()
class ModernBertFFN(keras.layers.Layer):
    """
    Feed-Forward Network with GeGLU activation for ModernBERT.
    ... (rest of the class is correct) ...
    """

    def __init__(self, config: ModernBertConfig, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config = config
        self.wi = None
        self.activation = None
        self.dropout = None
        self.wo = None
        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        if self.built:
            return
        self._build_input_shape = input_shape
        self.wi = keras.layers.Dense(
            units=self.config.intermediate_size * 2,
            use_bias=self.config.use_bias,
            kernel_initializer=initializers.TruncatedNormal(stddev=self.config.initializer_range),
            name="wi_geglu"
        )
        self.activation = keras.layers.Activation(self.config.hidden_act, name="activation")
        self.dropout = keras.layers.Dropout(self.config.hidden_dropout_prob, name="dropout")
        self.wo = keras.layers.Dense(
            units=self.config.hidden_size,
            use_bias=self.config.use_bias,
            kernel_initializer=initializers.TruncatedNormal(stddev=self.config.initializer_range),
            name="wo"
        )
        self.wi.build(input_shape)
        intermediate_shape = input_shape[:-1] + (self.config.intermediate_size,)
        self.wo.build(intermediate_shape)
        super().build(input_shape)

    def call(self, hidden_states: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        gate_and_main = self.wi(hidden_states)
        gate, main = ops.split(gate_and_main, 2, axis=-1)
        hidden_states = self.activation(gate) * main
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.wo(hidden_states)
        return hidden_states

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({'config': self.config.to_dict()})
        return config

    def get_build_config(self) -> Dict[str, Any]:
        return {'input_shape': self._build_input_shape}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        if config.get('input_shape') is not None:
            self.build(config['input_shape'])

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ModernBertFFN':
        bert_config = ModernBertConfig.from_dict(config['config'])
        return cls(config=bert_config)


@keras.saving.register_keras_serializable()
class ModernBertTransformerLayer(keras.layers.Layer):
    """
    A single ModernBERT encoder layer with pre-normalization.
    ... (rest of the class is correct) ...
    """

    def __init__(self, config: ModernBertConfig, is_global: bool, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config = config
        self.is_global = is_global
        self.transformer_layer = None
        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        if self.built:
            return
        self._build_input_shape = input_shape
        self.transformer_layer = TransformerLayer(
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_heads,
            intermediate_size=self.config.intermediate_size * 2,
            attention_type='multi_head_attention',
            normalization_position='pre',
            ffn_type='glu',
            dropout_rate=self.config.hidden_dropout_prob,
            attention_dropout_rate=self.config.attention_probs_dropout_prob,
            activation=self.config.hidden_act,
            use_bias=self.config.use_bias,
            kernel_initializer=initializers.TruncatedNormal(stddev=self.config.initializer_range),
            name="transformer"
        )
        self.transformer_layer.build(input_shape)
        super().build(input_shape)

    def call(self, hidden_states: keras.KerasTensor, attention_mask: Optional[keras.KerasTensor] = None,
             training: Optional[bool] = None) -> keras.KerasTensor:
        return self.transformer_layer(hidden_states, attention_mask=attention_mask, training=training)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({'config': self.config.to_dict(), 'is_global': self.is_global})
        return config

    def get_build_config(self) -> Dict[str, Any]:
        return {'input_shape': self._build_input_shape}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        if config.get('input_shape') is not None:
            self.build(config['input_shape'])

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ModernBertTransformerLayer':
        bert_config = ModernBertConfig.from_dict(config['config'])
        return cls(config=bert_config, is_global=config['is_global'])


@keras.saving.register_keras_serializable()
class ModernBERT(keras.Model):
    """
    ModernBERT (A Modern Bidirectional Encoder) model implementation.
    ... (rest of the class is correct) ...
    """

    def __init__(
            self,
            config: ModernBertConfig,
            add_pooling_layer: bool = True,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        config.validate()
        self.config = config
        self.add_pooling_layer = add_pooling_layer
        self.embeddings = None
        self.encoder_layers = []
        self.final_norm = None
        self.pooler = None
        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        if self.built:
            return
        self._build_input_shape = input_shape
        if isinstance(input_shape, dict):
            sequence_shape = input_shape.get('input_ids', (None, None))
        else:
            sequence_shape = input_shape
        self.embeddings = ModernBertEmbeddings(self.config, name="embeddings")
        self.encoder_layers = []
        for i in range(self.config.num_layers):
            is_global = (i + 1) % self.config.global_attention_interval == 0
            layer = ModernBertTransformerLayer(
                config=self.config,
                is_global=is_global,
                name=f"encoder_layer_{i}"
            )
            self.encoder_layers.append(layer)

        # FIX: Update LayerNormalization to use `center` for Keras 3 API
        self.final_norm = keras.layers.LayerNormalization(
            epsilon=self.config.layer_norm_eps,
            center=self.config.use_bias,
            name="final_layer_norm"
        )

        if self.add_pooling_layer:
            self.pooler = keras.layers.Dense(
                units=self.config.hidden_size,
                activation="tanh",
                use_bias=self.config.use_bias,
                kernel_initializer=initializers.TruncatedNormal(stddev=self.config.initializer_range),
                name="pooler"
            )
        embedding_output_shape = tuple(sequence_shape) + (self.config.hidden_size,)
        self.embeddings.build(sequence_shape)
        layer_input_shape = embedding_output_shape
        for layer in self.encoder_layers:
            layer.build(layer_input_shape)
        self.final_norm.build(embedding_output_shape)
        if self.pooler is not None:
            pooler_input_shape = (None, self.config.hidden_size)
            self.pooler.build(pooler_input_shape)
        super().build(input_shape)

    def call(self, inputs, attention_mask=None, token_type_ids=None, training=None, return_dict=False):
        if isinstance(inputs, dict):
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask", attention_mask)
            token_type_ids = inputs.get("token_type_ids", token_type_ids)
        else:
            input_ids = inputs
        if input_ids is None:
            raise ValueError("input_ids must be provided")

        embedding_output = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids, training=training)
        hidden_states = embedding_output
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask, training=training)

        sequence_output = self.final_norm(hidden_states, training=training)
        pooled_output = None
        if self.pooler is not None:
            first_token_tensor = sequence_output[:, 0]
            pooled_output = self.pooler(first_token_tensor)

        if return_dict:
            outputs = {"last_hidden_state": sequence_output}
            if pooled_output is not None:
                outputs["pooler_output"] = pooled_output
            return outputs
        else:
            if pooled_output is not None:
                return sequence_output, pooled_output
            return sequence_output

    # ... get_config, from_config, etc. are correct ...
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'config': self.config.to_dict(),
            'add_pooling_layer': self.add_pooling_layer,
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        return {'input_shape': self._build_input_shape}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        if config.get('input_shape') is not None:
            self.build(config['input_shape'])

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ModernBERT':
        bert_config = ModernBertConfig.from_dict(config['config'])
        return cls(config=bert_config, add_pooling_layer=config.get('add_pooling_layer', True))

# Factory functions for common configurations
def create_modern_bert_base() -> ModernBertConfig:
    """
    Create configuration for ModernBERT-base model.

    Architecture: 22 layers, 768 hidden, 12 heads, 1152 intermediate

    Returns:
        ModernBertConfig for base model.
    """
    logger.info("Creating ModernBERT-base configuration")
    return ModernBertConfig(
        vocab_size=50368,
        hidden_size=768,
        num_layers=22,
        num_heads=12,
        intermediate_size=1152,
        hidden_act="gelu",
        use_bias=False,
        normalization_type="layer_norm",
        global_attention_interval=3,
        local_attention_window_size=128
    )


def create_modern_bert_large() -> ModernBertConfig:
    """
    Create configuration for ModernBERT-large model.

    Architecture: 28 layers, 1024 hidden, 16 heads, 2624 intermediate

    Returns:
        ModernBertConfig for large model.
    """
    logger.info("Creating ModernBERT-large configuration")
    return ModernBertConfig(
        vocab_size=50368,
        hidden_size=1024,
        num_layers=28,
        num_heads=16,
        intermediate_size=2624,
        hidden_act="gelu",
        use_bias=False,
        normalization_type="layer_norm",
        global_attention_interval=3,
        local_attention_window_size=128
    )


def create_modern_bert_for_classification(
        config: ModernBertConfig,
        num_labels: int,
        classifier_dropout: Optional[float] = None
) -> keras.Model:
    """
    Create a ModernBERT model with a classification head.

    Args:
        config: ModernBertConfig for the base model.
        num_labels: Number of classification labels.
        classifier_dropout: Dropout rate for classifier. Uses config default if None.

    Returns:
        Compiled Keras model ready for classification training.

    Example:
        ```python
        config = create_modern_bert_base()
        model = create_modern_bert_for_classification(config, num_labels=10)

        # Compile model
        model.compile(
            optimizer='adamw',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        ```
    """
    logger.info(f"Creating ModernBERT classification model with {num_labels} labels")

    # Define inputs
    input_ids = keras.layers.Input(shape=(None,), dtype="int32", name="input_ids")
    attention_mask = keras.layers.Input(shape=(None,), dtype="int32", name="attention_mask")
    token_type_ids = keras.layers.Input(shape=(None,), dtype="int32", name="token_type_ids")

    # Create base ModernBERT model with pooling
    modern_bert = ModernBERT(config=config, add_pooling_layer=True, name="modern_bert")

    # Get ModernBERT outputs
    bert_outputs = modern_bert(
        inputs={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        },
        return_dict=True
    )
    pooled_output = bert_outputs["pooler_output"]

    # Apply classifier dropout
    if classifier_dropout is None:
        classifier_dropout = config.classifier_dropout or config.hidden_dropout_prob

    if classifier_dropout > 0.0:
        pooled_output = keras.layers.Dropout(
            classifier_dropout, name="classifier_dropout"
        )(pooled_output)

    # Final classification layer
    logits = keras.layers.Dense(
        units=num_labels,
        kernel_initializer=initializers.TruncatedNormal(
            stddev=config.initializer_range
        ),
        name="classifier"
    )(pooled_output)

    # Create final model
    model = keras.Model(
        inputs=[input_ids, attention_mask, token_type_ids],
        outputs=logits,
        name="modern_bert_for_classification"
    )

    logger.info(f"Created ModernBERT classification model with {model.count_params()} parameters")
    return model


def create_modern_bert_for_sequence_output(
        config: ModernBertConfig
) -> keras.Model:
    """
    Create a ModernBERT model for sequence-level tasks (e.g., token classification).

    Args:
        config: ModernBertConfig for the model.

    Returns:
        Keras model outputting sequence representations.

    Example:
        ```python
        config = create_modern_bert_base()
        model = create_modern_bert_for_sequence_output(config)
        ```
    """
    logger.info("Creating ModernBERT model for sequence output")

    # Define inputs
    input_ids = keras.layers.Input(shape=(None,), dtype="int32", name="input_ids")
    attention_mask = keras.layers.Input(shape=(None,), dtype="int32", name="attention_mask")
    token_type_ids = keras.layers.Input(shape=(None,), dtype="int32", name="token_type_ids")

    # Create base ModernBERT model without pooling
    modern_bert = ModernBERT(config=config, add_pooling_layer=False, name="modern_bert")

    # Get sequence output
    sequence_output = modern_bert(
        inputs={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
    )

    # Create final model
    model = keras.Model(
        inputs=[input_ids, attention_mask, token_type_ids],
        outputs=sequence_output,
        name="modern_bert_for_sequence_output"
    )

    logger.info(f"Created ModernBERT sequence model with {model.count_params()} parameters")
    return model