"""
NLP Task Head Factory

A comprehensive factory for building configurable head networks for various NLP tasks.
Designed to be model-agnostic and work with any NLP foundation model (BERT, GPT, T5, etc.).
"""

import keras
from keras import layers, ops
from typing import Dict, List, Optional, Union, Tuple, Any, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..activations import ActivationType
from ..standard_blocks import DenseBlock
from ..ffn import create_ffn_layer, FFNType
from ..attention import create_attention_layer, AttentionType
from ..norms import create_normalization_layer, NormalizationType

from .task_types import NLPTaskType, NLPTaskConfig

# ---------------------------------------------------------------------
# Base NLP Head Class
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class BaseNLPHead(keras.layers.Layer):
    """
    Base class for all NLP task heads.

    Provides common functionality and structure for task-specific heads,
    designed to work with any NLP foundation model's output.

    Args:
        task_config: NLPTaskConfig object with task configuration
        input_dim: Dimension of input features from foundation model
        normalization_type: Type of normalization to use
        activation_type: Type of activation function
        use_pooling: Whether to use pooling for sequence-level tasks
        pooling_type: Type of pooling ('mean', 'max', 'cls', 'attention')
        use_intermediate: Whether to use intermediate dense layer
        intermediate_size: Size of intermediate layer if used
        use_task_attention: Whether to use task-specific attention
        attention_type: Type of attention mechanism if used
        use_ffn: Whether to include FFN block
        ffn_type: Type of FFN to use
        ffn_expansion_factor: Expansion factor for FFN
        initializer_range: Standard deviation for weight initialization
        **kwargs: Additional arguments for base Layer class
    """

    def __init__(
            self,
            task_config: NLPTaskConfig,
            input_dim: int,
            normalization_type: NormalizationType = 'layer_norm',
            activation_type: ActivationType = 'gelu',
            use_pooling: bool = True,
            pooling_type: Literal['mean', 'max', 'cls', 'attention'] = 'cls',
            use_intermediate: bool = True,
            intermediate_size: Optional[int] = None,
            use_task_attention: bool = False,
            attention_type: AttentionType = 'multi_head',
            use_ffn: bool = False,
            ffn_type: FFNType = 'mlp',
            ffn_expansion_factor: int = 4,
            initializer_range: float = 0.02,
            **kwargs: Any
    ) -> None:
        """Initialize the base NLP head."""
        # Set a default name ONLY if 'name' is not in kwargs. ---
        # This prevents passing 'name' twice during deserialization, as the
        # saved config will already contain it.
        kwargs.setdefault('name', f"{task_config.name}_head")
        super().__init__(**kwargs)

        # Store configuration
        self.task_config = task_config
        self.input_dim = input_dim
        self.normalization_type = normalization_type
        self.activation_type = activation_type
        self.use_pooling = use_pooling
        self.pooling_type = pooling_type
        self.use_intermediate = use_intermediate
        self.intermediate_size = intermediate_size or input_dim
        self.use_task_attention = use_task_attention
        self.attention_type = attention_type
        self.use_ffn = use_ffn
        self.ffn_type = ffn_type
        self.ffn_expansion_factor = ffn_expansion_factor
        self.initializer_range = initializer_range

        # Set hidden size from config or use intermediate size
        self.hidden_size = task_config.hidden_size or self.intermediate_size

        # Create common layers (following Golden Rule: CREATE in __init__)
        self._create_common_layers()

    def _create_common_layers(self) -> None:
        """Create common layers used across different heads."""

        # Dropout layer
        self.dropout = layers.Dropout(
            self.task_config.dropout_rate,
            name=f"{self.name}_dropout"
        )

        # Optional normalization
        self.norm = create_normalization_layer(
            self.normalization_type,
            name=f"{self.name}_norm"
        )

        # Optional pooling for sequence-level tasks
        self.attention_pooling = None
        if self.use_pooling and self.pooling_type == 'attention':
            # Attention-based pooling
            self.attention_pooling = layers.Dense(
                1,
                activation='tanh',
                kernel_initializer=keras.initializers.TruncatedNormal(
                    stddev=self.initializer_range
                ),
                name=f"{self.name}_attention_pooling"
            )

        # Optional task-specific attention
        self.task_attention = None
        if self.use_task_attention:
            # Map attention parameters correctly
            attn_params = {
                'dropout_rate': self.task_config.dropout_rate,
                'name': f"{self.name}_attention"
            }

            if self.attention_type == 'multi_head':
                attn_params['dim'] = self.hidden_size
                attn_params['num_heads'] = 8
            else:
                attn_params['dim'] = self.hidden_size
                if self.attention_type in ['window', 'sliding_window']:
                    attn_params['window_size'] = 7
                if self.attention_type in ['multi_head', 'window', 'sliding_window']:
                    attn_params['num_heads'] = 8

            self.task_attention = create_attention_layer(
                self.attention_type,
                **attn_params
            )

        # Optional intermediate layer
        self.intermediate = None
        if self.use_intermediate:
            self.intermediate = DenseBlock(
                units=self.hidden_size,
                normalization_type=self.normalization_type,
                activation_type=self.activation_type,
                dropout_rate=self.task_config.dropout_rate,
                name=f"{self.name}_intermediate"
            )

        # Optional FFN block
        self.ffn = None
        if self.use_ffn:
            self.ffn = create_ffn_layer(
                self.ffn_type,
                hidden_dim=self.hidden_size * self.ffn_expansion_factor,
                output_dim=self.hidden_size,
                dropout_rate=self.task_config.dropout_rate,
                name=f"{self.name}_ffn"
            )

    def _pool_sequence(
            self,
            sequence: keras.KerasTensor,
            attention_mask: Optional[keras.KerasTensor] = None
    ) -> keras.KerasTensor:
        """
        Pool sequence representations to a single vector.

        Args:
            sequence: Sequence tensor [batch_size, seq_length, hidden_dim]
            attention_mask: Optional attention mask [batch_size, seq_length]

        Returns:
            Pooled representation [batch_size, hidden_dim]
        """
        if self.pooling_type == 'cls':
            # Use first token (CLS token for BERT-like models)
            return sequence[:, 0, :]

        elif self.pooling_type == 'mean':
            # Mean pooling with attention mask
            if attention_mask is not None:
                mask = ops.cast(attention_mask, dtype=sequence.dtype)
                mask_expanded = ops.expand_dims(mask, axis=-1)
                sum_embeddings = ops.sum(sequence * mask_expanded, axis=1)
                sum_mask = ops.sum(mask_expanded, axis=1)
                return sum_embeddings / ops.maximum(sum_mask, 1e-9)
            else:
                return ops.mean(sequence, axis=1)

        elif self.pooling_type == 'max':
            # Max pooling
            if attention_mask is not None:
                mask = ops.cast(attention_mask, dtype=sequence.dtype)
                mask_expanded = ops.expand_dims(mask, axis=-1)
                # Set masked positions to very negative value
                sequence = sequence * mask_expanded + (1 - mask_expanded) * -1e9
            return ops.max(sequence, axis=1)

        elif self.pooling_type == 'attention':
            # Attention-based pooling
            attention_weights = self.attention_pooling(sequence)
            attention_weights = ops.squeeze(attention_weights, axis=-1)

            if attention_mask is not None:
                mask = ops.cast(attention_mask, dtype=attention_weights.dtype)
                attention_weights = attention_weights * mask + (1 - mask) * -1e9

            attention_weights = ops.softmax(attention_weights, axis=-1)
            attention_weights = ops.expand_dims(attention_weights, axis=-1)

            return ops.sum(sequence * attention_weights, axis=1)

        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

    def build(self, input_shape: Union[Tuple, Dict]) -> None:
        """Build the layer and its sub-layers."""
        # Determine input shape
        if isinstance(input_shape, dict):
            hidden_shape = input_shape.get('hidden_states', (None, None, self.input_dim))
        else:
            hidden_shape = input_shape

        # Build normalization layer
        if len(hidden_shape) == 3:  # Sequence input
            norm_input_shape = hidden_shape
        else:  # Already pooled
            norm_input_shape = (hidden_shape[0], self.input_dim)

        self.norm.build(norm_input_shape)

        # Build attention pooling if needed
        if self.attention_pooling is not None:
            self.attention_pooling.build(hidden_shape)

        # Build task attention if needed
        if self.task_attention is not None:
            # Task attention operates on normalized features
            self.task_attention.build((hidden_shape[0], hidden_shape[1], self.hidden_size))

        # Build intermediate layer if needed
        if self.intermediate is not None:
            # Intermediate can receive pooled or sequence input
            if self.use_pooling and len(hidden_shape) == 3:
                intermediate_input = (hidden_shape[0], self.input_dim)
            else:
                intermediate_input = norm_input_shape
            self.intermediate.build(intermediate_input)

        # Build FFN if needed
        if self.ffn is not None:
            ffn_input = (hidden_shape[0], hidden_shape[1] if len(hidden_shape) == 3 else 1, self.hidden_size)
            self.ffn.build(ffn_input)

        super().build(input_shape)

    def compute_output_shape(self, input_shape: Union[Tuple, Dict]) -> Dict[str, Tuple]:
        """Compute the output shape of the layer."""
        # Base implementation - subclasses override with specific shapes
        if isinstance(input_shape, dict):
            batch_size = input_shape.get('hidden_states', (None,))[0]
        else:
            batch_size = input_shape[0] if input_shape else None

        return {'output': (batch_size, None)}

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'task_config': {
                'name': self.task_config.name,
                'task_type': self.task_config.task_type.value,
                'num_classes': self.task_config.num_classes,
                'dropout_rate': self.task_config.dropout_rate,
                'hidden_size': self.task_config.hidden_size,
                'loss_weight': self.task_config.loss_weight,
                'label_smoothing': self.task_config.label_smoothing,
                'use_crf': self.task_config.use_crf,
                'use_attention_pooling': self.task_config.use_attention_pooling,
                'vocabulary_size': getattr(self.task_config, 'vocabulary_size', None),
            },
            'input_dim': self.input_dim,
            'normalization_type': self.normalization_type,
            'activation_type': self.activation_type,
            'use_pooling': self.use_pooling,
            'pooling_type': self.pooling_type,
            'use_intermediate': self.use_intermediate,
            'intermediate_size': self.intermediate_size,
            'use_task_attention': self.use_task_attention,
            'attention_type': self.attention_type,
            'use_ffn': self.use_ffn,
            'ffn_type': self.ffn_type,
            'ffn_expansion_factor': self.ffn_expansion_factor,
            'initializer_range': self.initializer_range,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseNLPHead":
        """Create layer from configuration."""
        # Reconstruct task config
        task_config_dict = config.pop('task_config')
        task_config = NLPTaskConfig(
            name=task_config_dict['name'],
            task_type=NLPTaskType(task_config_dict['task_type']),
            num_classes=task_config_dict.get('num_classes'),
            dropout_rate=task_config_dict.get('dropout_rate', 0.1),
            hidden_size=task_config_dict.get('hidden_size'),
            loss_weight=task_config_dict.get('loss_weight', 1.0),
            label_smoothing=task_config_dict.get('label_smoothing', 0.0),
            use_crf=task_config_dict.get('use_crf', False),
            use_attention_pooling=task_config_dict.get('use_attention_pooling', False),
            vocabulary_size=task_config_dict.get('vocabulary_size'),
        )
        config['task_config'] = task_config
        return cls(**config)


# ---------------------------------------------------------------------
# Text Classification Head
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class TextClassificationHead(BaseNLPHead):
    """
    Head for text classification tasks.

    Supports various classification scenarios including sentiment analysis,
    topic classification, intent detection, etc.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize text classification head."""
        super().__init__(**kwargs)

        if self.task_config.num_classes is None:
            raise ValueError("num_classes must be specified for classification tasks")

        # Classification layer (CREATE in __init__)
        self.classifier = layers.Dense(
            self.task_config.num_classes,
            kernel_initializer=keras.initializers.TruncatedNormal(
                stddev=self.initializer_range
            ),
            name=f"{self.name}_classifier"
        )

    def build(self, input_shape: Union[Tuple, Dict]) -> None:
        """Build the layer and its sub-layers."""
        # First build common layers
        super().build(input_shape)

        # The dimension depends on whether transformative layers are used.
        if self.use_ffn or self.use_task_attention or self.use_intermediate:
            classifier_input_dim = self.hidden_size
        else:
            classifier_input_dim = self.input_dim

        # Classifier receives features after processing
        batch_size = input_shape[0] if isinstance(input_shape, tuple) else \
            input_shape.get('hidden_states', (None,))[0]
        classifier_input_shape = (batch_size, classifier_input_dim)
        self.classifier.build(classifier_input_shape)

    def compute_output_shape(self, input_shape: Union[Tuple, Dict]) -> Dict[str, Tuple]:
        """Compute the output shape of the layer."""
        if isinstance(input_shape, dict):
            batch_size = input_shape.get('hidden_states', (None,))[0]
        else:
            batch_size = input_shape[0] if input_shape else None

        return {
            'logits': (batch_size, self.task_config.num_classes),
            'probabilities': (batch_size, self.task_config.num_classes)
        }

    def call(
            self,
            inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
            training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """
        Forward pass through classification head.

        Args:
            inputs: Either sequence tensor or dict with 'hidden_states' and optional 'attention_mask'
            training: Whether in training mode

        Returns:
            Dictionary with 'logits' and 'probabilities'
        """
        # Handle different input formats
        if isinstance(inputs, dict):
            hidden_states = inputs['hidden_states']
            attention_mask = inputs.get('attention_mask', None)
        else:
            hidden_states = inputs
            attention_mask = None

        # Pool if needed (for sequence-level classification)
        if len(ops.shape(hidden_states)) == 3:  # [batch, seq_len, hidden]
            hidden_states = self._pool_sequence(hidden_states, attention_mask)

        # Apply common processing
        hidden_states = self.norm(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)

        if self.use_intermediate:
            hidden_states = self.intermediate(hidden_states, training=training)

        if self.use_task_attention and len(ops.shape(hidden_states)) == 3:
            hidden_states = self.task_attention(hidden_states, training=training)

        if self.use_ffn:
            hidden_states = self.ffn(hidden_states, training=training)

        # Classification
        logits = self.classifier(hidden_states)
        probabilities = ops.softmax(logits, axis=-1)

        return {
            'logits': logits,
            'probabilities': probabilities
        }


# ---------------------------------------------------------------------
# Token Classification Head
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class TokenClassificationHead(BaseNLPHead):
    """
    Head for token-level classification tasks.

    Supports NER, POS tagging, and other token-level tasks.
    Optionally supports CRF layer for sequence labeling.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize token classification head."""
        # Token classification doesn't use pooling
        kwargs['use_pooling'] = False
        super().__init__(**kwargs)

        if self.task_config.num_classes is None:
            raise ValueError("num_classes must be specified for token classification")

        # Token classifier (CREATE in __init__)
        self.token_classifier = layers.Dense(
            self.task_config.num_classes,
            kernel_initializer=keras.initializers.TruncatedNormal(
                stddev=self.initializer_range
            ),
            name=f"{self.name}_token_classifier"
        )

        # Optional CRF layer (placeholder - would need actual CRF implementation)
        if self.task_config.use_crf:
            # Note: CRF would need to be implemented separately
            # This is a placeholder for the architecture
            self.use_crf = True
        else:
            self.use_crf = False

    def build(self, input_shape: Union[Tuple, Dict]) -> None:
        """Build the layer and its sub-layers."""
        # First build common layers
        super().build(input_shape)

        # Determine input shape for the classifier
        if isinstance(input_shape, dict):
            seq_shape = input_shape.get('hidden_states', (None, None, self.input_dim))
        else:
            seq_shape = input_shape

        # Determine the correct input dimension for the classifier
        if self.use_ffn or self.use_task_attention or self.use_intermediate:
            classifier_input_dim = self.hidden_size
        else:
            classifier_input_dim = self.input_dim

        # Classifier receives processed sequence
        classifier_input_shape = (seq_shape[0], seq_shape[1], classifier_input_dim)
        self.token_classifier.build(classifier_input_shape)

    def compute_output_shape(self, input_shape: Union[Tuple, Dict]) -> Dict[str, Tuple]:
        """Compute the output shape of the layer."""
        if isinstance(input_shape, dict):
            batch_size = input_shape.get('hidden_states', (None,))[0]
            seq_length = input_shape.get('hidden_states', (None, None))[1]
        else:
            batch_size = input_shape[0] if input_shape else None
            seq_length = input_shape[1] if len(input_shape) > 1 else None

        output_shapes = {
            'logits': (batch_size, seq_length, self.task_config.num_classes)
        }

        if not self.use_crf:
            output_shapes['predictions'] = (batch_size, seq_length)

        return output_shapes

    def call(
            self,
            inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
            training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """
        Forward pass through token classification head.

        Returns:
            Dictionary with 'logits' and optional 'predictions'
        """
        # Handle different input formats
        if isinstance(inputs, dict):
            sequence_output = inputs['hidden_states']
        else:
            sequence_output = inputs

        # Apply processing to each token
        hidden_states = self.norm(sequence_output)
        hidden_states = self.dropout(hidden_states, training=training)

        if self.use_intermediate:
            # Apply to each position in the sequence
            batch_size, seq_len, hidden_dim = ops.shape(hidden_states)
            hidden_states_flat = ops.reshape(hidden_states, (-1, hidden_dim))
            hidden_states_flat = self.intermediate(hidden_states_flat, training=training)
            hidden_states = ops.reshape(hidden_states_flat, (batch_size, seq_len, -1))

        if self.use_task_attention:
            hidden_states = self.task_attention(hidden_states, training=training)

        if self.use_ffn:
            hidden_states = self.ffn(hidden_states, training=training)

        # Token classification
        logits = self.token_classifier(hidden_states)

        outputs = {'logits': logits}

        if not self.use_crf:
            # Simple argmax predictions
            predictions = ops.argmax(logits, axis=-1)
            outputs['predictions'] = predictions

        return outputs


# ---------------------------------------------------------------------
# Question Answering Head
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class QuestionAnsweringHead(BaseNLPHead):
    """
    Head for extractive question answering.

    Predicts start and end positions for answer spans.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize QA head."""
        # QA doesn't use pooling
        kwargs['use_pooling'] = False
        super().__init__(**kwargs)

        # Start and end position predictors (CREATE in __init__)
        self.start_classifier = layers.Dense(
            1,
            kernel_initializer=keras.initializers.TruncatedNormal(
                stddev=self.initializer_range
            ),
            name=f"{self.name}_start"
        )

        self.end_classifier = layers.Dense(
            1,
            kernel_initializer=keras.initializers.TruncatedNormal(
                stddev=self.initializer_range
            ),
            name=f"{self.name}_end"
        )

    def build(self, input_shape: Union[Tuple, Dict]) -> None:
        """Build the layer and its sub-layers."""
        # First build common layers
        super().build(input_shape)

        # Determine input shape for the classifiers
        if isinstance(input_shape, dict):
            seq_shape = input_shape.get('hidden_states', (None, None, self.input_dim))
        else:
            seq_shape = input_shape

        # Determine the correct input dimension for the classifiers ---
        if self.use_ffn or self.use_task_attention or self.use_intermediate:
            classifier_input_dim = self.hidden_size
        else:
            classifier_input_dim = self.input_dim

        # Classifiers receive processed sequence
        classifier_input_shape = (seq_shape[0], seq_shape[1], classifier_input_dim)
        self.start_classifier.build(classifier_input_shape)
        self.end_classifier.build(classifier_input_shape)

    def compute_output_shape(self, input_shape: Union[Tuple, Dict]) -> Dict[str, Tuple]:
        """Compute the output shape of the layer."""
        if isinstance(input_shape, dict):
            batch_size = input_shape.get('hidden_states', (None,))[0]
            seq_length = input_shape.get('hidden_states', (None, None))[1]
        else:
            batch_size = input_shape[0] if input_shape else None
            seq_length = input_shape[1] if len(input_shape) > 1 else None

        return {
            'start_logits': (batch_size, seq_length),
            'end_logits': (batch_size, seq_length)
        }

    def call(
            self,
            inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
            training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """
        Forward pass through QA head.

        Returns:
            Dictionary with 'start_logits', 'end_logits', and optionally answer spans
        """
        # Handle different input formats
        if isinstance(inputs, dict):
            sequence_output = inputs['hidden_states']
            attention_mask = inputs.get('attention_mask', None)
        else:
            sequence_output = inputs
            attention_mask = None

        # Apply processing
        hidden_states = self.norm(sequence_output)
        hidden_states = self.dropout(hidden_states, training=training)

        if self.use_intermediate:
            batch_size, seq_len, hidden_dim = ops.shape(hidden_states)
            hidden_states_flat = ops.reshape(hidden_states, (-1, hidden_dim))
            hidden_states_flat = self.intermediate(hidden_states_flat, training=training)
            hidden_states = ops.reshape(hidden_states_flat, (batch_size, seq_len, -1))

        if self.use_task_attention:
            hidden_states = self.task_attention(hidden_states, training=training)

        # Predict start and end positions
        start_logits = ops.squeeze(self.start_classifier(hidden_states), axis=-1)
        end_logits = ops.squeeze(self.end_classifier(hidden_states), axis=-1)

        # Apply attention mask if provided
        if attention_mask is not None:
            # Cast attention_mask to the same dtype as logits
            mask = ops.cast(attention_mask, dtype=start_logits.dtype)
            start_logits = start_logits * mask + (1 - mask) * -1e9
            end_logits = end_logits * mask + (1 - mask) * -1e9

        return {
            'start_logits': start_logits,
            'end_logits': end_logits,
        }


# ---------------------------------------------------------------------
# Text Similarity Head
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class TextSimilarityHead(BaseNLPHead):
    """
    Head for text similarity and semantic matching tasks.

    Can output similarity scores or embeddings for comparison.
    """

    def __init__(
            self,
            output_embeddings: bool = True,
            similarity_function: Literal['cosine', 'dot', 'learned'] = 'cosine',
            **kwargs: Any
    ) -> None:
        """Initialize similarity head."""
        super().__init__(**kwargs)

        self.output_embeddings = output_embeddings
        self.similarity_function = similarity_function

        # Optional projection layer (CREATE in __init__)
        self.projection = layers.Dense(
            self.hidden_size,
            kernel_initializer=keras.initializers.TruncatedNormal(
                stddev=self.initializer_range
            ),
            name=f"{self.name}_projection"
        )

        # Learned similarity function layers
        self.similarity_layers = []
        if similarity_function == 'learned':
            self.similarity_layers = [
                layers.Dense(
                    self.hidden_size,
                    activation=self.activation_type,
                    name=f"{self.name}_sim_hidden"
                ),
                layers.Dense(1, name=f"{self.name}_sim_output")
            ]

    def build(self, input_shape: Union[Tuple, Dict, List]) -> None:
        """Build the layer and its sub-layers."""
        # Handle tuple input (pairwise) by using first element shape
        if isinstance(input_shape, (list, tuple)) and len(input_shape) == 2:
            base_input_shape = input_shape[0]
        else:
            base_input_shape = input_shape

        # First build common layers
        super().build(base_input_shape)

        # Determine the correct input dimension for the projection layer ---
        # Note: Similarity head doesn't typically use task_attention.
        if self.use_ffn or self.use_intermediate:
            projection_input_dim = self.hidden_size
        else:
            projection_input_dim = self.input_dim

        projection_input_shape = (None, projection_input_dim)
        self.projection.build(projection_input_shape)

        # Build similarity layers if needed
        if self.similarity_function == 'learned':
            # Combined features: emb1, emb2, emb1*emb2, abs(emb1-emb2)
            combined_input_shape = (None, self.hidden_size * 4)
            for layer in self.similarity_layers:
                layer.build(combined_input_shape)
                combined_input_shape = (None, layer.units if hasattr(layer, 'units') else 1)

    def compute_output_shape(self, input_shape: Union[Tuple, Dict, List]) -> Dict[str, Tuple]:
        """Compute the output shape of the layer."""
        # Handle different input formats
        if isinstance(input_shape, (list, tuple)) and len(input_shape) == 2:
            # Pairwise input
            if isinstance(input_shape[0], dict):
                batch_size = input_shape[0].get('hidden_states', (None,))[0]
            else:
                batch_size = input_shape[0][0] if input_shape[0] else None

            outputs = {'similarity_score': (batch_size,)}
            if self.output_embeddings:
                outputs['embeddings_1'] = (batch_size, self.hidden_size)
                outputs['embeddings_2'] = (batch_size, self.hidden_size)
            return outputs
        else:
            # Single input - return embeddings
            if isinstance(input_shape, dict):
                batch_size = input_shape.get('hidden_states', (None,))[0]
            else:
                batch_size = input_shape[0] if input_shape else None

            return {'embeddings': (batch_size, self.hidden_size)}

    def call(
            self,
            inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor], Tuple],
            training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """
        Forward pass through similarity head.

        Args:
            inputs: Can be:
                - Single sequence tensor
                - Dict with 'hidden_states'
                - Tuple of two sequences for pairwise similarity

        Returns:
            Dictionary with 'embeddings' and/or 'similarity_score'
        """
        # Handle different input formats
        if isinstance(inputs, tuple) and len(inputs) == 2:
            # Pairwise similarity
            seq1, seq2 = inputs
            if isinstance(seq1, dict):
                seq1 = seq1['hidden_states']
            if isinstance(seq2, dict):
                seq2 = seq2['hidden_states']

            # Pool sequences
            if len(ops.shape(seq1)) == 3:
                seq1 = self._pool_sequence(seq1)
                seq2 = self._pool_sequence(seq2)

            # Process sequences
            emb1 = self._process_sequence(seq1, training)
            emb2 = self._process_sequence(seq2, training)

            # Compute similarity
            if self.similarity_function == 'cosine':
                # Cosine similarity
                emb1_norm = emb1 / ops.maximum(ops.norm(emb1, axis=-1, keepdims=True), 1e-8)
                emb2_norm = emb2 / ops.maximum(ops.norm(emb2, axis=-1, keepdims=True), 1e-8)
                similarity = ops.sum(emb1_norm * emb2_norm, axis=-1)

            elif self.similarity_function == 'dot':
                # Dot product
                similarity = ops.sum(emb1 * emb2, axis=-1)

            elif self.similarity_function == 'learned':
                # Learned similarity
                combined = ops.concatenate([emb1, emb2, emb1 * emb2, ops.abs(emb1 - emb2)], axis=-1)
                for layer in self.similarity_layers:
                    combined = layer(combined)
                similarity = ops.squeeze(combined, axis=-1)

            outputs = {'similarity_score': similarity}
            if self.output_embeddings:
                outputs['embeddings_1'] = emb1
                outputs['embeddings_2'] = emb2

            return outputs

        else:
            # Single sequence - return embeddings
            if isinstance(inputs, dict):
                hidden_states = inputs['hidden_states']
            else:
                hidden_states = inputs

            if len(ops.shape(hidden_states)) == 3:
                hidden_states = self._pool_sequence(hidden_states)

            embeddings = self._process_sequence(hidden_states, training)

            return {'embeddings': embeddings}

    def _process_sequence(
            self,
            sequence: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Process a sequence through the head layers."""
        hidden_states = self.norm(sequence)
        hidden_states = self.dropout(hidden_states, training=training)

        if self.use_intermediate:
            hidden_states = self.intermediate(hidden_states, training=training)

        if self.use_ffn:
            hidden_states = self.ffn(hidden_states, training=training)

        # Project to embedding space
        embeddings = self.projection(hidden_states)

        return embeddings

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'output_embeddings': self.output_embeddings,
            'similarity_function': self.similarity_function,
        })
        return config


# ---------------------------------------------------------------------
# Text Generation Head
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class TextGenerationHead(BaseNLPHead):
    """
    Head for text generation tasks.

    Supports autoregressive generation, masked language modeling, etc.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize generation head."""
        # Generation doesn't use pooling
        kwargs['use_pooling'] = False
        super().__init__(**kwargs)

        if self.task_config.vocabulary_size is None:
            raise ValueError("vocabulary_size must be specified for generation tasks")

        # Language modeling head (CREATE in __init__)
        self.lm_head = layers.Dense(
            self.task_config.vocabulary_size,
            kernel_initializer=keras.initializers.TruncatedNormal(
                stddev=self.initializer_range
            ),
            name=f"{self.name}_lm_head"
        )

    def build(self, input_shape: Union[Tuple, Dict]) -> None:
        """Build the layer and its sub-layers."""
        # First build common layers
        super().build(input_shape)

        # Determine input shape for the LM head
        if isinstance(input_shape, dict):
            seq_shape = input_shape.get('hidden_states', (None, None, self.input_dim))
        else:
            seq_shape = input_shape

        # Determine the correct input dimension for the LM head ---
        if self.use_ffn or self.use_task_attention or self.use_intermediate:
            lm_input_dim = self.hidden_size
        else:
            lm_input_dim = self.input_dim

        # LM head receives processed sequence
        lm_input_shape = (seq_shape[0], seq_shape[1], lm_input_dim)
        self.lm_head.build(lm_input_shape)

    def compute_output_shape(self, input_shape: Union[Tuple, Dict]) -> Dict[str, Tuple]:
        """Compute the output shape of the layer."""
        if isinstance(input_shape, dict):
            batch_size = input_shape.get('hidden_states', (None,))[0]
            seq_length = input_shape.get('hidden_states', (None, None))[1]
        else:
            batch_size = input_shape[0] if input_shape else None
            seq_length = input_shape[1] if len(input_shape) > 1 else None

        return {'logits': (batch_size, seq_length, self.task_config.vocabulary_size)}

    def call(
            self,
            inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
            training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """
        Forward pass through generation head.

        Returns:
            Dictionary with 'logits' over vocabulary
        """
        # Handle different input formats
        if isinstance(inputs, dict):
            sequence_output = inputs['hidden_states']
        else:
            sequence_output = inputs

        # Apply processing
        hidden_states = self.norm(sequence_output)
        hidden_states = self.dropout(hidden_states, training=training)

        if self.use_intermediate:
            batch_size, seq_len, hidden_dim = ops.shape(hidden_states)
            hidden_states_flat = ops.reshape(hidden_states, (-1, hidden_dim))
            hidden_states_flat = self.intermediate(hidden_states_flat, training=training)
            hidden_states = ops.reshape(hidden_states_flat, (batch_size, seq_len, -1))

        if self.use_ffn:
            hidden_states = self.ffn(hidden_states, training=training)

        # Predict token logits
        logits = self.lm_head(hidden_states)

        return {'logits': logits}


# ---------------------------------------------------------------------
# Multiple Choice Head
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MultipleChoiceHead(BaseNLPHead):
    """
    Head for multiple choice tasks.

    Handles questions with multiple answer options.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize multiple choice head."""
        super().__init__(**kwargs)

        # Scorer for each choice (CREATE in __init__)
        self.scorer = layers.Dense(
            1,
            kernel_initializer=keras.initializers.TruncatedNormal(
                stddev=self.initializer_range
            ),
            name=f"{self.name}_scorer"
        )

    def build(self, input_shape: Union[Tuple, Dict]) -> None:
        """Build the layer and its sub-layers."""
        # First build common layers
        super().build(input_shape)

        # Determine the correct input dimension for the scorer ---
        if self.use_ffn or self.use_task_attention or self.use_intermediate:
            scorer_input_dim = self.hidden_size
        else:
            scorer_input_dim = self.input_dim

        # Scorer receives pooled representations for each choice
        scorer_input_shape = (None, scorer_input_dim)
        self.scorer.build(scorer_input_shape)

    def compute_output_shape(self, input_shape: Union[Tuple, Dict]) -> Dict[str, Tuple]:
        """Compute the output shape of the layer."""
        if isinstance(input_shape, dict):
            hidden_shape = input_shape.get('hidden_states', (None, None))
        else:
            hidden_shape = input_shape

        batch_size = hidden_shape[0] if hidden_shape else None
        num_choices = hidden_shape[1] if len(hidden_shape) > 1 else None

        return {
            'logits': (batch_size, num_choices),
            'probabilities': (batch_size, num_choices)
        }

    def call(
            self,
            inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
            training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """
        Forward pass through multiple choice head.

        Args:
            inputs: Should have shape [batch_size, num_choices, seq_len, hidden_dim]
                   or be a dict with such 'hidden_states'

        Returns:
            Dictionary with 'logits' over choices
        """
        if isinstance(inputs, dict):
            hidden_states = inputs['hidden_states']
        else:
            hidden_states = inputs

        # Reshape to handle multiple choices
        batch_size, num_choices = ops.shape(hidden_states)[:2]

        # Pool each choice if needed
        if len(ops.shape(hidden_states)) == 4:  # [batch, choices, seq, hidden]
            # Reshape to process all choices together
            hidden_states = ops.reshape(
                hidden_states,
                (batch_size * num_choices,) + ops.shape(hidden_states)[2:]
            )
            pooled = self._pool_sequence(hidden_states)
            pooled = ops.reshape(pooled, (batch_size, num_choices, -1))
        else:
            pooled = hidden_states

        # Process each choice
        hidden_states = self.norm(pooled)
        hidden_states = self.dropout(hidden_states, training=training)

        if self.use_intermediate:
            # Flatten for dense layer
            hidden_shape = ops.shape(hidden_states)
            hidden_flat = ops.reshape(hidden_states, (-1, hidden_shape[-1]))
            hidden_flat = self.intermediate(hidden_flat, training=training)
            hidden_states = ops.reshape(hidden_flat, hidden_shape)

        # Score each choice
        logits = self.scorer(hidden_states)
        logits = ops.squeeze(logits, axis=-1)

        return {
            'logits': logits,
            'probabilities': ops.softmax(logits, axis=-1)
        }


# ---------------------------------------------------------------------
# Multi-Task NLP Head
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MultiTaskNLPHead(keras.layers.Layer):
    """
    Multi-task head that combines multiple task-specific NLP heads.

    Args:
        task_configs: Dictionary mapping task names to NLPTaskConfig objects
        shared_input_dim: Dimension of shared features from foundation model
        use_task_specific_projections: Whether to use task-specific projections
        **kwargs: Additional arguments
    """

    def __init__(
            self,
            task_configs: Dict[str, NLPTaskConfig],
            shared_input_dim: int,
            use_task_specific_projections: bool = False,
            **kwargs: Any
    ) -> None:
        """Initialize multi-task head."""
        super().__init__(**kwargs)

        self.task_configs = task_configs
        self.shared_input_dim = shared_input_dim
        self.use_task_specific_projections = use_task_specific_projections

        # Create task heads and projections (CREATE in __init__)
        self.task_heads = {}
        self.task_projections = {}

        for task_name, task_config in task_configs.items():
            # Create appropriate head
            head_class = get_head_class(task_config.task_type)

            # Set input dimension
            if use_task_specific_projections:
                projection_dim = task_config.hidden_size or shared_input_dim
                self.task_projections[task_name] = layers.Dense(
                    projection_dim,
                    name=f"{task_name}_projection"
                )
                input_dim = projection_dim
            else:
                input_dim = shared_input_dim

            # Create head
            self.task_heads[task_name] = head_class(
                task_config=task_config,
                input_dim=input_dim
            )

    def build(self, input_shape: Union[Tuple, Dict]) -> None:
        """Build the layer and its sub-layers."""
        # Build task projections if needed
        if self.use_task_specific_projections:
            for task_name, projection in self.task_projections.items():
                if isinstance(input_shape, dict):
                    hidden_shape = input_shape.get('hidden_states', (None, None, self.shared_input_dim))
                else:
                    hidden_shape = input_shape

                # Projections work on flattened features
                if len(hidden_shape) == 3:
                    projection_input = (None, self.shared_input_dim)
                else:
                    projection_input = hidden_shape
                projection.build(projection_input)

        # Build task heads
        for task_name, head in self.task_heads.items():
            if self.use_task_specific_projections:
                # Head receives projected features
                task_config = self.task_configs[task_name]
                head_input_dim = task_config.hidden_size or self.shared_input_dim
                if isinstance(input_shape, dict):
                    head_input = {'hidden_states': (None, None, head_input_dim)}
                else:
                    head_input = (None, None, head_input_dim) if len(input_shape) == 3 else (None, head_input_dim)
            else:
                head_input = input_shape

            head.build(head_input)

        super().build(input_shape)

    def compute_output_shape(self, input_shape: Union[Tuple, Dict]) -> Dict[str, Dict[str, Tuple]]:
        """Compute the output shape of the layer."""
        output_shapes = {}

        for task_name, head in self.task_heads.items():
            # Get shape for each task head
            if self.use_task_specific_projections:
                task_config = self.task_configs[task_name]
                head_input_dim = task_config.hidden_size or self.shared_input_dim
                if isinstance(input_shape, dict):
                    hidden_states_shape = input_shape.get('hidden_states')
                    head_input = {'hidden_states': (hidden_states_shape[0],
                                                   hidden_states_shape[1],
                                                   head_input_dim)}
                else:
                    shape = (input_shape[0], input_shape[1], head_input_dim) if len(input_shape) == 3 else \
                        (input_shape[0], head_input_dim)
                    head_input = shape
            else:
                head_input = input_shape

            output_shapes[task_name] = head.compute_output_shape(head_input)

        return output_shapes

    def call(
            self,
            inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
            task_name: Optional[str] = None,
            training: Optional[bool] = None
    ) -> Union[Dict[str, Dict[str, keras.KerasTensor]], Dict[str, keras.KerasTensor]]:
        """
        Forward pass through multi-task head.

        Args:
            inputs: Input features from foundation model
            task_name: Specific task to run (if None, runs all tasks)
            training: Whether in training mode

        Returns:
            Dictionary of task outputs or single task output
        """
        # Handle single task
        if task_name is not None:
            if task_name not in self.task_heads:
                raise ValueError(f"Unknown task: {task_name}")

            task_inputs = inputs

            # Apply task-specific projection if needed
            if self.use_task_specific_projections:
                if isinstance(task_inputs, dict):
                    hidden_states = task_inputs['hidden_states']
                    hidden_states = self.task_projections[task_name](hidden_states)
                    task_inputs = {**task_inputs, 'hidden_states': hidden_states}
                else:
                    task_inputs = self.task_projections[task_name](task_inputs)

            return self.task_heads[task_name](task_inputs, training=training)

        # Run all tasks
        outputs = {}
        for task_name, task_head in self.task_heads.items():
            task_inputs = inputs

            # Apply task-specific projection if needed
            if self.use_task_specific_projections:
                if isinstance(task_inputs, dict):
                    hidden_states = task_inputs['hidden_states']
                    hidden_states = self.task_projections[task_name](hidden_states)
                    task_inputs = {**task_inputs, 'hidden_states': hidden_states}
                else:
                    task_inputs = self.task_projections[task_name](task_inputs)

            outputs[task_name] = task_head(task_inputs, training=training)

        return outputs

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'task_configs': {
                name: {
                    'name': tc.name,
                    'task_type': tc.task_type.value,
                    'num_classes': tc.num_classes,
                    'dropout_rate': tc.dropout_rate,
                    'hidden_size': tc.hidden_size,
                    'vocabulary_size': getattr(tc, 'vocabulary_size', None),
                }
                for name, tc in self.task_configs.items()
            },
            'shared_input_dim': self.shared_input_dim,
            'use_task_specific_projections': self.use_task_specific_projections,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MultiTaskNLPHead":
        """Create layer from configuration."""
        # Reconstruct task configs
        task_configs_dict = config.pop('task_configs')
        task_configs = {}

        for name, tc_dict in task_configs_dict.items():
            task_configs[name] = NLPTaskConfig(
                name=tc_dict['name'],
                task_type=NLPTaskType(tc_dict['task_type']),
                num_classes=tc_dict.get('num_classes'),
                dropout_rate=tc_dict.get('dropout_rate', 0.1),
                hidden_size=tc_dict.get('hidden_size'),
                vocabulary_size=tc_dict.get('vocabulary_size'),
            )

        config['task_configs'] = task_configs
        return cls(**config)


# ---------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------

def get_head_class(task_type: NLPTaskType) -> type:
    """Get the appropriate head class for a task type."""
    head_mapping = {
        # Classification tasks
        NLPTaskType.TEXT_CLASSIFICATION: TextClassificationHead,
        NLPTaskType.SENTIMENT_ANALYSIS: TextClassificationHead,
        NLPTaskType.EMOTION_DETECTION: TextClassificationHead,
        NLPTaskType.INTENT_CLASSIFICATION: TextClassificationHead,
        NLPTaskType.TOPIC_CLASSIFICATION: TextClassificationHead,
        NLPTaskType.SPAM_DETECTION: TextClassificationHead,

        # Token classification
        NLPTaskType.TOKEN_CLASSIFICATION: TokenClassificationHead,
        NLPTaskType.NAMED_ENTITY_RECOGNITION: TokenClassificationHead,
        NLPTaskType.PART_OF_SPEECH_TAGGING: TokenClassificationHead,
        NLPTaskType.SEQUENCE_LABELING: TokenClassificationHead,

        # QA and span tasks
        NLPTaskType.QUESTION_ANSWERING: QuestionAnsweringHead,
        NLPTaskType.SPAN_EXTRACTION: QuestionAnsweringHead,

        # Similarity tasks
        NLPTaskType.TEXT_SIMILARITY: TextSimilarityHead,
        NLPTaskType.PARAPHRASE_DETECTION: TextSimilarityHead,
        NLPTaskType.DUPLICATE_DETECTION: TextSimilarityHead,

        # Generation tasks
        NLPTaskType.TEXT_GENERATION: TextGenerationHead,
        NLPTaskType.MASKED_LANGUAGE_MODELING: TextGenerationHead,
        NLPTaskType.TEXT_SUMMARIZATION: TextGenerationHead,
        NLPTaskType.TEXT_COMPLETION: TextGenerationHead,

        # Multiple choice
        NLPTaskType.MULTIPLE_CHOICE: MultipleChoiceHead,

        # NLI can use classification head with 3 classes
        NLPTaskType.NATURAL_LANGUAGE_INFERENCE: TextClassificationHead,

        # Regression tasks use classification head with num_classes=1
        NLPTaskType.TEXT_REGRESSION: TextClassificationHead,
        NLPTaskType.READABILITY_SCORING: TextClassificationHead,
        NLPTaskType.QUALITY_SCORING: TextClassificationHead,
    }

    return head_mapping.get(task_type, TextClassificationHead)


def create_nlp_head(
        task_config: Union[NLPTaskConfig, Dict[str, Any]],
        input_dim: int,
        **kwargs: Any
) -> BaseNLPHead:
    """
    Factory function to create NLP task heads.

    Args:
        task_config: NLPTaskConfig object or dict with task configuration
        input_dim: Dimension of input features from foundation model
        **kwargs: Additional configuration parameters

    Returns:
        Configured NLP head for the specified task

    Example:
        >>> # Create sentiment classification head
        >>> config = NLPTaskConfig(
        ...     name="sentiment",
        ...     task_type=NLPTaskType.SENTIMENT_ANALYSIS,
        ...     num_classes=3
        ... )
        >>> head = create_nlp_head(config, input_dim=768)

        >>> # Create NER head with CRF
        >>> ner_config = NLPTaskConfig(
        ...     name="ner",
        ...     task_type=NLPTaskType.NAMED_ENTITY_RECOGNITION,
        ...     num_classes=9,
        ...     use_crf=True
        ... )
        >>> ner_head = create_nlp_head(ner_config, input_dim=768)
    """
    # Convert dict to NLPTaskConfig if needed
    if isinstance(task_config, dict):
        task_config = NLPTaskConfig(**task_config)

    # Get appropriate head class
    head_class = get_head_class(task_config.task_type)

    # Create head with configuration
    return head_class(
        task_config=task_config,
        input_dim=input_dim,
        **kwargs
    )


def create_multi_task_nlp_head(
        task_configs: Union[List[NLPTaskConfig], Dict[str, NLPTaskConfig]],
        input_dim: int,
        **kwargs: Any
) -> MultiTaskNLPHead:
    """
    Create a multi-task NLP head from task configurations.

    Args:
        task_configs: List or dict of NLPTaskConfig objects
        input_dim: Dimension of input features from foundation model
        **kwargs: Additional configuration

    Returns:
        MultiTaskNLPHead instance

    Example:
        >>> # Create multi-task head for GLUE-like tasks
        >>> configs = [
        ...     NLPTaskConfig("cola", NLPTaskType.TEXT_CLASSIFICATION, num_classes=2),
        ...     NLPTaskConfig("sst", NLPTaskType.SENTIMENT_ANALYSIS, num_classes=5),
        ...     NLPTaskConfig("mrpc", NLPTaskType.PARAPHRASE_DETECTION, num_classes=2),
        ...     NLPTaskConfig("sts", NLPTaskType.TEXT_SIMILARITY),
        ... ]
        >>> multi_head = create_multi_task_nlp_head(configs, input_dim=768)
    """
    # Convert list to dict if needed
    if isinstance(task_configs, list):
        task_configs = {config.name: config for config in task_configs}

    return MultiTaskNLPHead(
        task_configs=task_configs,
        shared_input_dim=input_dim,
        **kwargs
    )


# ---------------------------------------------------------------------
# Configuration Helpers
# ---------------------------------------------------------------------

class NLPHeadConfiguration:
    """Configuration helper for NLP heads."""

    @staticmethod
    def get_default_config(task_type: NLPTaskType) -> Dict[str, Any]:
        """Get default configuration for a task type."""
        base_config = {
            'dropout_rate': 0.1,
            'normalization_type': 'layer_norm',
            'activation_type': 'gelu',
            'use_intermediate': True,
            'initializer_range': 0.02,
        }

        task_specific = {
            NLPTaskType.TEXT_CLASSIFICATION: {
                'use_pooling': True,
                'pooling_type': 'cls',
                'use_task_attention': False,
            },
            NLPTaskType.TOKEN_CLASSIFICATION: {
                'use_pooling': False,
                'use_crf': False,
                'use_task_attention': False,
            },
            NLPTaskType.QUESTION_ANSWERING: {
                'use_pooling': False,
                'use_task_attention': True,
                'attention_type': 'multi_head',
            },
            NLPTaskType.TEXT_SIMILARITY: {
                'use_pooling': True,
                'pooling_type': 'mean',
                'output_embeddings': True,
                'similarity_function': 'cosine',
            },
            NLPTaskType.TEXT_GENERATION: {
                'use_pooling': False,
                'vocabulary_size': 32000,
                'use_ffn': True,
                'ffn_type': 'swiglu',
            },
        }

        config = base_config.copy()
        if task_type in task_specific:
            config.update(task_specific[task_type])

        return config