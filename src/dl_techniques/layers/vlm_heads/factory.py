"""
VLM Task Head Factory

A comprehensive factory for building configurable head networks for Visual Language Model tasks.
Designed to be model-agnostic and work with any VLM foundation model (CLIP, BLIP, Flamingo, etc.).
"""

import keras
from keras import layers, ops
from typing import Dict, List, Optional, Union, Tuple, Any, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..activations import ActivationType
from ..ffn.factory import create_ffn_layer, FFNType
from ..attention import create_attention_layer, AttentionType
from ..norms import create_normalization_layer, NormalizationType

from .task_types import VLMTaskType, VLMTaskConfig


# ---------------------------------------------------------------------
# Multi-Modal Fusion Modules
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MultiModalFusion(keras.layers.Layer):
    """
    Multi-modal fusion layer for combining vision and language features.

    Args:
        fusion_type: Type of fusion ('concat', 'add', 'multiply', 'attention', 'gated')
        hidden_dim: Hidden dimension for fusion
        dropout_rate: Dropout rate
        use_layer_norm: Whether to use layer normalization
        **kwargs: Additional arguments
    """

    def __init__(
            self,
            fusion_type: Literal['concat', 'add', 'multiply', 'attention', 'gated'] = 'attention',
            hidden_dim: int = 768,
            dropout_rate: float = 0.1,
            use_layer_norm: bool = True,
            **kwargs: Any
    ) -> None:
        """Initialize multi-modal fusion."""
        super().__init__(**kwargs)

        self.fusion_type = fusion_type
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm

    def build(self, input_shape: Union[Tuple, List[Tuple]]) -> None:
        """Build the fusion layer."""
        super().build(input_shape)

        if isinstance(input_shape, list):
            vision_shape, text_shape = input_shape
            vision_dim = vision_shape[-1]
            text_dim = text_shape[-1]
        else:
            vision_dim = text_dim = input_shape[-1]

        if self.fusion_type == 'concat':
            self.fusion_proj = layers.Dense(
                self.hidden_dim,
                name=f"{self.name}_fusion_proj"
            )
        elif self.fusion_type == 'attention':
            # Cross-attention for fusion
            self.cross_attention = create_attention_layer(
                'multi_head',
                dim=self.hidden_dim,
                num_heads=8,
                dropout_rate=self.dropout_rate,
                name=f"{self.name}_cross_attention"
            )
            self.vision_proj = layers.Dense(self.hidden_dim, name=f"{self.name}_vision_proj")
            self.text_proj = layers.Dense(self.hidden_dim, name=f"{self.name}_text_proj")
        elif self.fusion_type == 'gated':
            # Gated fusion mechanism
            self.vision_gate = layers.Dense(
                self.hidden_dim,
                activation='sigmoid',
                name=f"{self.name}_vision_gate"
            )
            self.text_gate = layers.Dense(
                self.hidden_dim,
                activation='sigmoid',
                name=f"{self.name}_text_gate"
            )
            self.vision_proj = layers.Dense(self.hidden_dim, name=f"{self.name}_vision_proj")
            self.text_proj = layers.Dense(self.hidden_dim, name=f"{self.name}_text_proj")

        if self.use_layer_norm:
            self.layer_norm = layers.LayerNormalization(name=f"{self.name}_norm")

        if self.dropout_rate > 0:
            self.dropout = layers.Dropout(self.dropout_rate)

    def call(
            self,
            inputs: Union[Tuple[keras.KerasTensor, keras.KerasTensor], keras.KerasTensor],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through fusion layer."""
        if isinstance(inputs, (list, tuple)):
            vision_features, text_features = inputs
        else:
            # Assume concatenated features
            return inputs

        if self.fusion_type == 'concat':
            fused = ops.concatenate([vision_features, text_features], axis=-1)
            fused = self.fusion_proj(fused)

        elif self.fusion_type == 'add':
            # Ensure same dimensionality
            fused = vision_features + text_features

        elif self.fusion_type == 'multiply':
            # Element-wise multiplication
            fused = vision_features * text_features

        elif self.fusion_type == 'attention':
            # Cross-attention fusion
            vision_proj = self.vision_proj(vision_features)
            text_proj = self.text_proj(text_features)

            # Vision attends to text
            v2t = self.cross_attention(
                ops.concatenate([vision_proj, text_proj], axis=1),
                training=training
            )
            fused = v2t

        elif self.fusion_type == 'gated':
            # Gated fusion
            vision_proj = self.vision_proj(vision_features)
            text_proj = self.text_proj(text_features)

            vision_gate = self.vision_gate(text_features)
            text_gate = self.text_gate(vision_features)

            fused = vision_gate * vision_proj + text_gate * text_proj

        if self.use_layer_norm:
            fused = self.layer_norm(fused)

        if self.dropout_rate > 0:
            fused = self.dropout(fused, training=training)

        return fused

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'fusion_type': self.fusion_type,
            'hidden_dim': self.hidden_dim,
            'dropout_rate': self.dropout_rate,
            'use_layer_norm': self.use_layer_norm,
        })
        return config


# ---------------------------------------------------------------------
# Base VLM Head Class
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class BaseVLMHead(keras.layers.Layer):
    """
    Base class for all VLM task heads.

    Provides common functionality for multi-modal tasks.

    Args:
        task_config: VLMTaskConfig object with task configuration
        vision_dim: Dimension of vision features
        text_dim: Dimension of text features
        fusion_type: Type of multi-modal fusion
        normalization_type: Type of normalization to use
        activation_type: Type of activation function
        use_cross_attention: Whether to use cross-modal attention
        attention_type: Type of attention mechanism
        use_ffn: Whether to include FFN block
        ffn_type: Type of FFN to use
        ffn_expansion_factor: Expansion factor for FFN
        **kwargs: Additional arguments
    """

    def __init__(
            self,
            task_config: VLMTaskConfig,
            vision_dim: int = 768,
            text_dim: int = 768,
            fusion_type: Literal['concat', 'add', 'multiply', 'attention', 'gated'] = 'attention',
            normalization_type: NormalizationType = 'layer_norm',
            activation_type: ActivationType = 'gelu',
            use_cross_attention: bool = True,
            attention_type: AttentionType = 'multi_head',
            use_ffn: bool = True,
            ffn_type: FFNType = 'mlp',
            ffn_expansion_factor: int = 4,
            **kwargs: Any
    ) -> None:
        """Initialize base VLM head."""
        super().__init__(name=f"{task_config.name}_head", **kwargs)

        # Store configuration
        self.task_config = task_config
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.fusion_type = fusion_type or task_config.fusion_type
        self.normalization_type = normalization_type
        self.activation_type = activation_type
        self.use_cross_attention = use_cross_attention or task_config.use_cross_attention
        self.attention_type = attention_type
        self.use_ffn = use_ffn
        self.ffn_type = ffn_type
        self.ffn_expansion_factor = ffn_expansion_factor

        # Set hidden dimensions
        self.hidden_dim = task_config.hidden_size or max(vision_dim, text_dim)
        self.fusion_hidden_dim = task_config.fusion_hidden_size or self.hidden_dim

        # Build common layers
        self._build_common_layers()

    def _build_common_layers(self) -> None:
        """Build common layers used across different heads."""

        # Multi-modal fusion
        self.fusion = MultiModalFusion(
            fusion_type=self.fusion_type,
            hidden_dim=self.fusion_hidden_dim,
            dropout_rate=self.task_config.dropout_rate,
            name=f"{self.name}_fusion"
        )

        # Optional cross-modal attention
        if self.use_cross_attention:
            self.cross_attention = create_attention_layer(
                self.attention_type,
                dim=self.hidden_dim,
                num_heads=8 if self.attention_type == 'multi_head' else None,
                dropout_rate=self.task_config.dropout_rate,
                name=f"{self.name}_cross_attention"
            )

        # Normalization
        self.norm = create_normalization_layer(
            self.normalization_type,
            name=f"{self.name}_norm"
        )

        # Optional FFN
        if self.use_ffn:
            self.ffn = create_ffn_layer(
                self.ffn_type,
                hidden_dim=self.hidden_dim * self.ffn_expansion_factor,
                output_dim=self.hidden_dim,
                dropout_rate=self.task_config.dropout_rate,
                name=f"{self.name}_ffn"
            )

        # Dropout
        self.dropout = layers.Dropout(
            self.task_config.dropout_rate,
            name=f"{self.name}_dropout"
        )

    def build(self, input_shape: Union[Tuple, Dict]) -> None:
        """Build the layer."""
        super().build(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'task_config': {
                'name': self.task_config.name,
                'task_type': self.task_config.task_type.value,
                'vocab_size': self.task_config.vocab_size,
                'max_text_length': self.task_config.max_text_length,
                'hidden_size': self.task_config.hidden_size,
                'dropout_rate': self.task_config.dropout_rate,
            },
            'vision_dim': self.vision_dim,
            'text_dim': self.text_dim,
            'fusion_type': self.fusion_type,
            'normalization_type': self.normalization_type,
            'activation_type': self.activation_type,
            'use_cross_attention': self.use_cross_attention,
            'attention_type': self.attention_type,
            'use_ffn': self.use_ffn,
            'ffn_type': self.ffn_type,
            'ffn_expansion_factor': self.ffn_expansion_factor,
        })
        return config


# ---------------------------------------------------------------------
# Image Captioning Head
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ImageCaptioningHead(BaseVLMHead):
    """
    Head for image captioning tasks.

    Generates natural language descriptions of images.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize image captioning head."""
        super().__init__(**kwargs)

        # Decoder layers
        self.decoder_embedding = layers.Embedding(
            self.task_config.vocab_size,
            self.hidden_dim,
            name=f"{self.name}_decoder_embedding"
        )

        self.decoder_lstm = layers.LSTM(
            self.hidden_dim,
            return_sequences=True,
            name=f"{self.name}_decoder_lstm"
        )

        # Output projection
        self.output_proj = layers.Dense(
            self.task_config.vocab_size,
            name=f"{self.name}_output_proj"
        )

    def call(
            self,
            inputs: Dict[str, keras.KerasTensor],
            training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """
        Forward pass through captioning head.

        Args:
            inputs: Dictionary with 'vision_features' and optional 'text_input'
            training: Whether in training mode

        Returns:
            Dictionary with 'logits' and 'caption'
        """
        vision_features = inputs['vision_features']
        text_input = inputs.get('text_input', None)

        # Pool vision features if needed
        if len(ops.shape(vision_features)) == 3:  # [batch, patches, dim]
            if self.task_config.pooling_type == 'avg':
                vision_pooled = ops.mean(vision_features, axis=1)
            elif self.task_config.pooling_type == 'max':
                vision_pooled = ops.max(vision_features, axis=1)
            else:  # cls
                vision_pooled = vision_features[:, 0, :]
        else:
            vision_pooled = vision_features

        # Process vision features
        vision_processed = self.norm(vision_pooled)
        vision_processed = self.dropout(vision_processed, training=training)

        if self.use_ffn:
            vision_processed = self.ffn(vision_processed, training=training)

        # Initialize decoder with vision features
        batch_size = ops.shape(vision_features)[0]

        if text_input is not None:
            # Teacher forcing during training
            text_embedded = self.decoder_embedding(text_input)

            # Expand vision features for each time step
            seq_len = ops.shape(text_input)[1]
            vision_expanded = ops.expand_dims(vision_processed, axis=1)
            vision_expanded = ops.tile(vision_expanded, [1, seq_len, 1])

            # Combine vision and text
            decoder_input = self.fusion([vision_expanded, text_embedded], training=training)

            # Decode
            decoder_output = self.decoder_lstm(decoder_input, training=training)
        else:
            # Inference mode - autoregressive generation
            # Start with vision features
            decoder_output = ops.expand_dims(vision_processed, axis=1)
            decoder_output = self.decoder_lstm(decoder_output, training=training)

        # Project to vocabulary
        logits = self.output_proj(decoder_output)

        return {
            'logits': logits,
            'hidden_states': decoder_output
        }


# ---------------------------------------------------------------------
# Visual Question Answering Head
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class VQAHead(BaseVLMHead):
    """
    Head for visual question answering tasks.

    Answers questions about image content.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize VQA head."""
        super().__init__(**kwargs)

        # Question encoder
        self.question_encoder = layers.LSTM(
            self.hidden_dim,
            return_sequences=False,
            name=f"{self.name}_question_encoder"
        )

        # Answer decoder (can be classification or generation)
        self.answer_type_classifier = layers.Dense(
            3,  # yes/no, number, other
            activation='softmax',
            name=f"{self.name}_answer_type"
        )

        # For open-ended answers
        self.answer_generator = layers.Dense(
            self.task_config.vocab_size,
            name=f"{self.name}_answer_generator"
        )

        # For multiple choice
        if self.task_config.num_classes:
            self.answer_classifier = layers.Dense(
                self.task_config.num_classes,
                activation='softmax',
                name=f"{self.name}_answer_classifier"
            )

    def call(
            self,
            inputs: Dict[str, keras.KerasTensor],
            training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """
        Forward pass through VQA head.

        Args:
            inputs: Dictionary with 'vision_features' and 'question_features'
            training: Whether in training mode

        Returns:
            Dictionary with answer predictions
        """
        vision_features = inputs['vision_features']
        question_features = inputs['question_features']

        # Encode question
        if len(ops.shape(question_features)) == 3:  # Sequential features
            question_encoded = self.question_encoder(question_features, training=training)
        else:
            question_encoded = question_features

        # Pool vision features if needed
        if len(ops.shape(vision_features)) == 3:
            if self.task_config.pooling_type == 'avg':
                vision_pooled = ops.mean(vision_features, axis=1)
            else:
                vision_pooled = vision_features[:, 0, :]
        else:
            vision_pooled = vision_features

        # Fuse modalities
        fused = self.fusion([vision_pooled, question_encoded], training=training)

        # Apply cross-attention if enabled
        if self.use_cross_attention:
            fused = self.cross_attention(fused, training=training)

        # Process through FFN
        if self.use_ffn:
            fused = self.ffn(fused, training=training)

        # Generate outputs
        outputs = {
            'answer_type': self.answer_type_classifier(fused)
        }

        if self.task_config.num_classes:
            outputs['answer_logits'] = self.answer_classifier(fused)
        else:
            outputs['answer_tokens'] = self.answer_generator(fused)

        return outputs


# ---------------------------------------------------------------------
# Visual Grounding Head
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class VisualGroundingHead(BaseVLMHead):
    """
    Head for visual grounding tasks.

    Localizes image regions from text descriptions.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize visual grounding head."""
        super().__init__(**kwargs)

        # Grounding layers
        self.grounding_attention = create_attention_layer(
            'multi_head',
            dim=self.hidden_dim,
            num_heads=8,
            dropout_rate=self.task_config.dropout_rate,
            name=f"{self.name}_grounding_attention"
        )

        # Bounding box regression
        self.bbox_regressor = layers.Dense(
            4,  # [x1, y1, x2, y2]
            name=f"{self.name}_bbox_regressor"
        )

        # Confidence score
        self.confidence_scorer = layers.Dense(
            1,
            activation='sigmoid',
            name=f"{self.name}_confidence"
        )

    def call(
            self,
            inputs: Dict[str, keras.KerasTensor],
            training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """
        Forward pass through grounding head.

        Args:
            inputs: Dictionary with 'vision_features' and 'text_features'
            training: Whether in training mode

        Returns:
            Dictionary with bounding boxes and confidence scores
        """
        vision_features = inputs['vision_features']
        text_features = inputs['text_features']

        # Encode text query
        if len(ops.shape(text_features)) == 3:
            text_encoded = ops.mean(text_features, axis=1)
        else:
            text_encoded = text_features

        # Apply grounding attention
        # Text query attends to visual regions
        if len(ops.shape(vision_features)) == 3:
            # Patch/region features
            batch_size, num_regions, vision_dim = ops.shape(vision_features)

            # Expand text for each region
            text_expanded = ops.expand_dims(text_encoded, axis=1)
            text_expanded = ops.tile(text_expanded, [1, num_regions, 1])

            # Compute attention weights
            attended = self.grounding_attention(
                ops.concatenate([vision_features, text_expanded], axis=-1),
                training=training
            )

            # Get region scores
            region_scores = self.confidence_scorer(attended)
            region_scores = ops.squeeze(region_scores, axis=-1)

            # Get top scoring region
            top_idx = ops.argmax(region_scores, axis=1)
            batch_indices = ops.arange(batch_size)
            top_features = attended[batch_indices, top_idx]
        else:
            # Global features
            fused = self.fusion([vision_features, text_encoded], training=training)
            top_features = fused
            region_scores = self.confidence_scorer(fused)

        # Predict bounding box
        bbox = self.bbox_regressor(top_features)

        # Normalize bbox to [0, 1]
        bbox = ops.sigmoid(bbox)

        return {
            'bbox': bbox,
            'confidence': region_scores,
            'grounded_features': top_features
        }


# ---------------------------------------------------------------------
# Image-Text Matching Head
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ImageTextMatchingHead(BaseVLMHead):
    """
    Head for image-text matching tasks.

    Determines correspondence between images and text.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize matching head."""
        super().__init__(**kwargs)

        # Projection layers
        self.vision_proj = layers.Dense(
            self.hidden_dim,
            name=f"{self.name}_vision_proj"
        )

        self.text_proj = layers.Dense(
            self.hidden_dim,
            name=f"{self.name}_text_proj"
        )

        # Similarity computation
        self.similarity_head = layers.Dense(
            1,
            activation='sigmoid',
            name=f"{self.name}_similarity"
        )

        # Optional: contrastive head for retrieval
        self.temperature = self.add_weight(
            name=f"{self.name}_temperature",
            shape=(1,),
            initializer=keras.initializers.Constant(0.07),
            trainable=True
        )

    def call(
            self,
            inputs: Dict[str, keras.KerasTensor],
            training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """
        Forward pass through matching head.

        Args:
            inputs: Dictionary with 'vision_features' and 'text_features'
            training: Whether in training mode

        Returns:
            Dictionary with matching scores
        """
        vision_features = inputs['vision_features']
        text_features = inputs['text_features']

        # Pool features if needed
        if len(ops.shape(vision_features)) == 3:
            vision_pooled = ops.mean(vision_features, axis=1)
        else:
            vision_pooled = vision_features

        if len(ops.shape(text_features)) == 3:
            text_pooled = ops.mean(text_features, axis=1)
        else:
            text_pooled = text_features

        # Project to common space
        vision_proj = self.vision_proj(vision_pooled)
        text_proj = self.text_proj(text_pooled)

        # Normalize for contrastive learning
        vision_norm = vision_proj / ops.maximum(
            ops.norm(vision_proj, axis=-1, keepdims=True), 1e-8
        )
        text_norm = text_proj / ops.maximum(
            ops.norm(text_proj, axis=-1, keepdims=True), 1e-8
        )

        # Compute similarity matrix
        similarity_matrix = ops.matmul(vision_norm, ops.transpose(text_norm)) / self.temperature

        # Also compute element-wise matching score
        fused = self.fusion([vision_proj, text_proj], training=training)
        match_score = self.similarity_head(fused)

        return {
            'similarity_matrix': similarity_matrix,
            'match_score': ops.squeeze(match_score, axis=-1),
            'vision_embeddings': vision_proj,
            'text_embeddings': text_proj
        }


# ---------------------------------------------------------------------
# Dense Captioning Head
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class DenseCaptioningHead(BaseVLMHead):
    """
    Head for dense captioning tasks.

    Generates region-specific captions for image areas.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize dense captioning head."""
        super().__init__(**kwargs)

        # Region proposal network (simplified)
        self.region_proposer = layers.Dense(
            4,  # bbox coordinates
            name=f"{self.name}_region_proposer"
        )

        # Region classifier
        self.region_classifier = layers.Dense(
            2,  # object/no-object
            activation='softmax',
            name=f"{self.name}_region_classifier"
        )

        # Caption generator per region
        self.caption_generator = layers.LSTM(
            self.hidden_dim,
            return_sequences=True,
            name=f"{self.name}_caption_generator"
        )

        self.caption_proj = layers.Dense(
            self.task_config.vocab_size,
            name=f"{self.name}_caption_proj"
        )

    def call(
            self,
            inputs: Dict[str, keras.KerasTensor],
            training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """
        Forward pass through dense captioning head.

        Args:
            inputs: Dictionary with 'vision_features'
            training: Whether in training mode

        Returns:
            Dictionary with regions and captions
        """
        vision_features = inputs['vision_features']

        if len(ops.shape(vision_features)) != 3:
            # Need spatial features
            raise ValueError("Dense captioning requires spatial vision features")

        batch_size, num_patches, feature_dim = ops.shape(vision_features)

        # Propose regions
        region_bboxes = self.region_proposer(vision_features)
        region_scores = self.region_classifier(vision_features)

        # Select top regions
        objectness = region_scores[:, :, 1]  # Object class
        top_k = 10  # Number of regions to caption

        # Get top-k indices
        top_indices = ops.argsort(objectness, axis=1)[:, -top_k:]

        # Extract features for top regions
        batch_indices = ops.arange(batch_size)[:, None]
        batch_indices = ops.tile(batch_indices, [1, top_k])

        # Gather top region features
        top_features = ops.take_along_axis(vision_features, top_indices[:, :, None], axis=1)
        top_bboxes = ops.take_along_axis(region_bboxes, top_indices[:, :, None], axis=1)

        # Generate captions for each region
        # Initialize with region features
        captions_hidden = self.caption_generator(top_features, training=training)
        caption_logits = self.caption_proj(captions_hidden)

        return {
            'region_bboxes': ops.sigmoid(top_bboxes),  # Normalize to [0, 1]
            'region_scores': ops.take_along_axis(objectness, top_indices, axis=1),
            'caption_logits': caption_logits,
            'num_regions': top_k
        }


# ---------------------------------------------------------------------
# Visual Dialogue Head
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class VisualDialogueHead(BaseVLMHead):
    """
    Head for visual dialogue tasks.

    Multi-turn conversations about images.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize visual dialogue head."""
        super().__init__(**kwargs)

        # Dialogue encoder
        self.dialogue_encoder = layers.LSTM(
            self.hidden_dim,
            return_sequences=True,
            return_state=True,
            name=f"{self.name}_dialogue_encoder"
        )

        # Response decoder
        self.response_decoder = layers.LSTM(
            self.hidden_dim,
            return_sequences=True,
            name=f"{self.name}_response_decoder"
        )

        # Output projection
        self.output_proj = layers.Dense(
            self.task_config.vocab_size,
            name=f"{self.name}_output_proj"
        )

        # Optional: response ranker for retrieval-based dialogue
        self.response_ranker = layers.Dense(
            1,
            activation='sigmoid',
            name=f"{self.name}_response_ranker"
        )

    def call(
            self,
            inputs: Dict[str, keras.KerasTensor],
            training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """
        Forward pass through dialogue head.

        Args:
            inputs: Dictionary with 'vision_features', 'dialogue_history', 'current_utterance'
            training: Whether in training mode

        Returns:
            Dictionary with response
        """
        vision_features = inputs['vision_features']
        dialogue_history = inputs.get('dialogue_history', None)
        current_utterance = inputs['current_utterance']

        # Pool vision features
        if len(ops.shape(vision_features)) == 3:
            vision_pooled = ops.mean(vision_features, axis=1)
        else:
            vision_pooled = vision_features

        # Encode dialogue history
        if dialogue_history is not None:
            dialogue_encoded, state_h, state_c = self.dialogue_encoder(
                dialogue_history,
                training=training
            )
            context_state = [state_h, state_c]
        else:
            # Initialize with vision features
            context_state = [vision_pooled, vision_pooled]
            dialogue_encoded = None

        # Fuse current utterance with vision
        fused = self.fusion([vision_pooled, current_utterance], training=training)

        # Generate response
        response_hidden = self.response_decoder(
            ops.expand_dims(fused, axis=1),
            initial_state=context_state,
            training=training
        )

        # Project to vocabulary
        response_logits = self.output_proj(response_hidden)

        outputs = {
            'response_logits': response_logits,
            'dialogue_state': response_hidden
        }

        # Optional: rank candidate responses
        if 'candidate_responses' in inputs:
            candidates = inputs['candidate_responses']
            # Score each candidate
            scores = []
            for i in range(ops.shape(candidates)[1]):
                candidate = candidates[:, i, :]
                score = self.response_ranker(
                    self.fusion([fused, candidate], training=training)
                )
                scores.append(score)
            outputs['ranking_scores'] = ops.concatenate(scores, axis=1)

        return outputs


# ---------------------------------------------------------------------
# Multi-Task VLM Head
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MultiTaskVLMHead(keras.layers.Layer):
    """
    Multi-task head that combines multiple VLM task-specific heads.

    Args:
        task_configs: Dictionary mapping task names to VLMTaskConfig objects
        shared_vision_dim: Dimension of shared vision features
        shared_text_dim: Dimension of shared text features
        use_task_specific_projections: Whether to use task-specific projections
        **kwargs: Additional arguments
    """

    def __init__(
            self,
            task_configs: Dict[str, VLMTaskConfig],
            shared_vision_dim: int = 768,
            shared_text_dim: int = 768,
            use_task_specific_projections: bool = False,
            **kwargs: Any
    ) -> None:
        """Initialize multi-task VLM head."""
        super().__init__(**kwargs)

        self.task_configs = task_configs
        self.shared_vision_dim = shared_vision_dim
        self.shared_text_dim = shared_text_dim
        self.use_task_specific_projections = use_task_specific_projections

        # Create task heads
        self.task_heads = {}
        self.task_projections = {}

        for task_name, task_config in task_configs.items():
            # Create appropriate head
            head_class = get_head_class(task_config.task_type)

            # Set dimensions
            if use_task_specific_projections:
                vision_dim = task_config.vision_hidden_size or shared_vision_dim
                text_dim = task_config.text_hidden_size or shared_text_dim

                self.task_projections[task_name] = {
                    'vision': layers.Dense(vision_dim, name=f"{task_name}_vision_proj"),
                    'text': layers.Dense(text_dim, name=f"{task_name}_text_proj")
                }
            else:
                vision_dim = shared_vision_dim
                text_dim = shared_text_dim

            # Create head
            self.task_heads[task_name] = head_class(
                task_config=task_config,
                vision_dim=vision_dim,
                text_dim=text_dim
            )

    def call(
            self,
            inputs: Dict[str, keras.KerasTensor],
            task_name: Optional[str] = None,
            training: Optional[bool] = None
    ) -> Union[Dict[str, Dict[str, keras.KerasTensor]], Dict[str, keras.KerasTensor]]:
        """
        Forward pass through multi-task head.

        Args:
            inputs: Dictionary with vision and text features
            task_name: Specific task to run (if None, runs all tasks)
            training: Whether in training mode

        Returns:
            Dictionary of task outputs or single task output
        """
        # Handle single task
        if task_name is not None:
            if task_name not in self.task_heads:
                raise ValueError(f"Unknown task: {task_name}")

            task_inputs = inputs.copy()

            # Apply task-specific projections if needed
            if self.use_task_specific_projections:
                if 'vision_features' in task_inputs:
                    task_inputs['vision_features'] = self.task_projections[task_name]['vision'](
                        task_inputs['vision_features']
                    )
                if 'text_features' in task_inputs:
                    task_inputs['text_features'] = self.task_projections[task_name]['text'](
                        task_inputs['text_features']
                    )

            return self.task_heads[task_name](task_inputs, training=training)

        # Run all tasks
        outputs = {}
        for task_name, task_head in self.task_heads.items():
            task_inputs = inputs.copy()

            # Apply task-specific projections if needed
            if self.use_task_specific_projections:
                if 'vision_features' in task_inputs:
                    task_inputs['vision_features'] = self.task_projections[task_name]['vision'](
                        task_inputs['vision_features']
                    )
                if 'text_features' in task_inputs:
                    task_inputs['text_features'] = self.task_projections[task_name]['text'](
                        task_inputs['text_features']
                    )

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
                    'vocab_size': tc.vocab_size,
                    'hidden_size': tc.hidden_size,
                }
                for name, tc in self.task_configs.items()
            },
            'shared_vision_dim': self.shared_vision_dim,
            'shared_text_dim': self.shared_text_dim,
            'use_task_specific_projections': self.use_task_specific_projections,
        })
        return config


# ---------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------

def get_head_class(task_type: VLMTaskType) -> type:
    """Get the appropriate head class for a VLM task type."""
    head_mapping = {
        # Captioning tasks
        VLMTaskType.IMAGE_CAPTIONING: ImageCaptioningHead,
        VLMTaskType.DENSE_CAPTIONING: DenseCaptioningHead,
        VLMTaskType.VISUAL_STORYTELLING: ImageCaptioningHead,
        VLMTaskType.IMAGE_PARAGRAPH_CAPTIONING: ImageCaptioningHead,

        # VQA tasks
        VLMTaskType.VISUAL_QUESTION_ANSWERING: VQAHead,
        VLMTaskType.VISUAL_REASONING: VQAHead,
        VLMTaskType.VISUAL_COMMONSENSE_REASONING: VQAHead,
        VLMTaskType.CHART_QUESTION_ANSWERING: VQAHead,

        # Grounding tasks
        VLMTaskType.VISUAL_GROUNDING: VisualGroundingHead,
        VLMTaskType.REFERRING_EXPRESSION_COMPREHENSION: VisualGroundingHead,
        VLMTaskType.PHRASE_GROUNDING: VisualGroundingHead,

        # Matching tasks
        VLMTaskType.IMAGE_TEXT_MATCHING: ImageTextMatchingHead,
        VLMTaskType.IMAGE_RETRIEVAL: ImageTextMatchingHead,
        VLMTaskType.TEXT_RETRIEVAL: ImageTextMatchingHead,

        # Dialogue tasks
        VLMTaskType.VISUAL_DIALOGUE: VisualDialogueHead,
        VLMTaskType.VISUAL_CHAT: VisualDialogueHead,

        # Can extend with more specialized heads as needed
    }

    return head_mapping.get(task_type, BaseVLMHead)


def create_vlm_head(
        task_config: Union[VLMTaskConfig, Dict[str, Any]],
        vision_dim: int = 768,
        text_dim: int = 768,
        **kwargs: Any
) -> BaseVLMHead:
    """
    Factory function to create VLM task heads.

    Args:
        task_config: VLMTaskConfig object or dict with task configuration
        vision_dim: Dimension of vision features
        text_dim: Dimension of text features
        **kwargs: Additional configuration parameters

    Returns:
        Configured VLM head for the specified task

    Example:
        >>> # Create image captioning head
        >>> config = VLMTaskConfig(
        ...     name="captioning",
        ...     task_type=VLMTaskType.IMAGE_CAPTIONING,
        ...     vocab_size=50000
        ... )
        >>> head = create_vlm_head(config, vision_dim=768, text_dim=768)

        >>> # Create VQA head
        >>> vqa_config = VLMTaskConfig(
        ...     name="vqa",
        ...     task_type=VLMTaskType.VISUAL_QUESTION_ANSWERING
        ... )
        >>> vqa_head = create_vlm_head(vqa_config)
    """
    # Convert dict to VLMTaskConfig if needed
    if isinstance(task_config, dict):
        task_config = VLMTaskConfig(**task_config)

    # Get appropriate head class
    head_class = get_head_class(task_config.task_type)

    # Create head with configuration
    return head_class(
        task_config=task_config,
        vision_dim=vision_dim,
        text_dim=text_dim,
        **kwargs
    )


def create_multi_task_vlm_head(
        task_configs: Union[List[VLMTaskConfig], Dict[str, VLMTaskConfig]],
        vision_dim: int = 768,
        text_dim: int = 768,
        **kwargs: Any
) -> MultiTaskVLMHead:
    """
    Create a multi-task VLM head from task configurations.

    Args:
        task_configs: List or dict of VLMTaskConfig objects
        vision_dim: Dimension of vision features
        text_dim: Dimension of text features
        **kwargs: Additional configuration

    Returns:
        MultiTaskVLMHead instance

    Example:
        >>> # Create multi-task head for vision-language understanding
        >>> configs = [
        ...     VLMTaskConfig("caption", VLMTaskType.IMAGE_CAPTIONING),
        ...     VLMTaskConfig("vqa", VLMTaskType.VISUAL_QUESTION_ANSWERING),
        ...     VLMTaskConfig("grounding", VLMTaskType.VISUAL_GROUNDING),
        ... ]
        >>> multi_head = create_multi_task_vlm_head(configs)
    """
    # Convert list to dict if needed
    if isinstance(task_configs, list):
        task_configs = {config.name: config for config in task_configs}

    return MultiTaskVLMHead(
        task_configs=task_configs,
        shared_vision_dim=vision_dim,
        shared_text_dim=text_dim,
        **kwargs
    )


# ---------------------------------------------------------------------
# Configuration Helpers
# ---------------------------------------------------------------------

class VLMHeadConfiguration:
    """Configuration helper for VLM heads."""

    @staticmethod
    def get_default_config(task_type: VLMTaskType) -> Dict[str, Any]:
        """Get default configuration for a VLM task type."""
        base_config = {
            'dropout_rate': 0.1,
            'normalization_type': 'layer_norm',
            'activation_type': 'gelu',
            'use_cross_attention': True,
            'use_ffn': True,
            'ffn_type': 'mlp',
            'ffn_expansion_factor': 4,
        }

        task_specific = {
            VLMTaskType.IMAGE_CAPTIONING: {
                'fusion_type': 'attention',
                'vocab_size': 50000,
                'max_text_length': 100,
            },
            VLMTaskType.VISUAL_QUESTION_ANSWERING: {
                'fusion_type': 'attention',
                'vocab_size': 50000,
                'num_classes': None,  # Open-ended by default
            },
            VLMTaskType.VISUAL_GROUNDING: {
                'fusion_type': 'attention',
                'use_cross_attention': True,
            },
            VLMTaskType.IMAGE_TEXT_MATCHING: {
                'fusion_type': 'concat',
                'use_cross_attention': False,
            },
            VLMTaskType.VISUAL_DIALOGUE: {
                'fusion_type': 'gated',
                'vocab_size': 50000,
                'max_text_length': 512,
            },
        }

        config = base_config.copy()
        if task_type in task_specific:
            config.update(task_specific[task_type])

        return config