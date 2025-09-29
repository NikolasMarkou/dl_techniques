"""
VLM Task Head Factory

A comprehensive factory for building configurable head networks for Visual Language
Model tasks. Designed to be model-agnostic and work with any VLM foundation
model (CLIP, BLIP, Flamingo, etc.).
"""

import keras
from keras import layers, ops
from typing import Dict, List, Optional, Union, Tuple, Any

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from ..activations import ActivationType
from ..attention.factory import create_attention_layer
from ..ffn.factory import create_ffn_from_config, FFNType, create_ffn_layer
from ..fusion.multimodal_fusion import FusionStrategy, MultiModalFusion
from ..norms import NormalizationType
from ..norms.factory import create_normalization_layer
from .task_types import VLMTaskConfig, VLMTaskType


# ---------------------------------------------------------------------
# Base VLM Head Class
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class BaseVLMHead(keras.layers.Layer):
    """
    Base class for all VLM task heads, using an advanced fusion module.

    Provides common functionality for multi-modal tasks, delegating complex
    fusion logic to the dedicated MultiModalFusion layer.

    Args:
        task_config: VLMTaskConfig object with task configuration.
        vision_dim: Dimension of vision features.
        text_dim: Dimension of text features.
        fusion_strategy: The fusion strategy for the MultiModalFusion layer.
        fusion_config: A dictionary of configuration parameters for the
                       MultiModalFusion layer.
        normalization_type: Type of normalization for post-fusion blocks.
        activation_type: Type of activation function for post-fusion blocks.
        use_post_fusion_ffn: If True, includes an FFN block after fusion.
        ffn_type: Type of FFN to use in the post-fusion block.
        ffn_expansion_factor: Expansion factor for the post-fusion FFN.
        **kwargs: Additional arguments for the base Layer.
    """

    def __init__(
        self,
        task_config: VLMTaskConfig,
        vision_dim: int = 768,
        text_dim: int = 768,
        fusion_strategy: FusionStrategy = "cross_attention",
        fusion_config: Optional[Dict[str, Any]] = None,
        normalization_type: NormalizationType = "layer_norm",
        activation_type: ActivationType = "gelu",
        use_post_fusion_ffn: bool = True,
        ffn_type: FFNType = "mlp",
        ffn_expansion_factor: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=f"{task_config.name}_head", **kwargs)

        # Store configuration
        self.task_config = task_config
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.fusion_strategy = fusion_strategy
        self.fusion_config = fusion_config or {}
        self.normalization_type = normalization_type
        self.activation_type = activation_type
        self.use_post_fusion_ffn = use_post_fusion_ffn
        self.ffn_type = ffn_type
        self.ffn_expansion_factor = ffn_expansion_factor

        self.hidden_dim = task_config.hidden_size or max(vision_dim, text_dim)
        self._build_common_layers()

    def _build_common_layers(self) -> None:
        """Builds common layers used across different heads."""
        self.fusion = MultiModalFusion(
            dim=self.hidden_dim,
            fusion_strategy=self.fusion_strategy,
            dropout_rate=self.task_config.dropout_rate,
            name=f"{self.name}_fusion",
            **self.fusion_config,
        )

        # These blocks process the output of the fusion layer.
        self.post_fusion_norm = create_normalization_layer(
            self.normalization_type, name=f"{self.name}_post_fusion_norm"
        )
        self.post_fusion_dropout = layers.Dropout(
            self.task_config.dropout_rate, name=f"{self.name}_post_fusion_dropout"
        )

        if self.use_post_fusion_ffn:
            self.post_fusion_ffn = create_ffn_layer(
                self.ffn_type,
                hidden_dim=self.hidden_dim * self.ffn_expansion_factor,
                output_dim=self.hidden_dim,
                dropout_rate=self.task_config.dropout_rate,
                name=f"{self.name}_post_fusion_ffn",
            )

    def build(self, input_shape: Union[Tuple, Dict]) -> None:
        """Builds the layer."""
        super().build(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Gets layer configuration."""
        config = super().get_config()
        config.update(
            {
                "task_config": self.task_config.__dict__,
                "vision_dim": self.vision_dim,
                "text_dim": self.text_dim,
                "fusion_strategy": self.fusion_strategy,
                "fusion_config": self.fusion_config,
                "normalization_type": self.normalization_type,
                "activation_type": self.activation_type,
                "use_post_fusion_ffn": self.use_post_fusion_ffn,
                "ffn_type": self.ffn_type,
                "ffn_expansion_factor": self.ffn_expansion_factor,
            }
        )
        return config


# ---------------------------------------------------------------------
# Image Captioning Head
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ImageCaptioningHead(keras.layers.Layer):
    """
    An autoregressive decoder head for generating text conditioned on vision features.

    This head implements a multi-layer Transformer decoder, a standard
    architecture for sequence-to-sequence tasks, specifically adapted for image
    captioning. Its purpose is to generate a descriptive text sequence one token
    at a time, conditioned on a set of static visual features extracted from an
    image encoder. It uses self-attention to model the text and cross-attention
    to incorporate visual information at each layer.

    Args:
        task_config (VLMTaskConfig): Configuration object for the task.
        vision_dim (int): Dimension of vision features.
        text_dim (int): Dimension of text features.
        num_layers (int): Number of decoder layers. Defaults to 6.
        num_heads (int): Number of attention heads. Defaults to 12.
        ffn_type (FFNType): Type of feed-forward network in decoder blocks. Defaults to "swiglu".
        **kwargs: Additional arguments for the base Layer.
    """

    def __init__(
        self,
        task_config: VLMTaskConfig,
        vision_dim: int = 768,
        text_dim: int = 768,
        num_layers: int = 6,
        num_heads: int = 12,
        ffn_type: FFNType = "swiglu",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=f"{task_config.name}_head", **kwargs)
        self.task_config = task_config
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.hidden_dim = task_config.hidden_size or text_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_type = ffn_type

        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )

        # CREATE all sub-layers in __init__
        self.cross_attention_layers = []
        self.self_attention_layers = []
        self.ffn_layers = []
        self.norm_layers = []

        for i in range(self.num_layers):
            cross_attn = create_attention_layer(
                "multi_head",
                dim=self.hidden_dim,
                num_heads=self.num_heads,
                dropout_rate=self.task_config.dropout_rate,
                name=f"cross_attention_{i}",
            )
            self.cross_attention_layers.append(cross_attn)

            self_attn = create_attention_layer(
                "multi_head",
                dim=self.hidden_dim,
                num_heads=self.num_heads,
                dropout_rate=self.task_config.dropout_rate,
                name=f"self_attention_{i}",
            )
            self.self_attention_layers.append(self_attn)

            ffn = create_ffn_from_config(
                {"type": self.ffn_type, "output_dim": self.hidden_dim, "name": f"ffn_{i}"}
            )
            self.ffn_layers.append(ffn)

            norm1 = create_normalization_layer("rms_norm", name=f"norm1_{i}")
            norm2 = create_normalization_layer("rms_norm", name=f"norm2_{i}")
            norm3 = create_normalization_layer("rms_norm", name=f"norm3_{i}")
            self.norm_layers.extend([norm1, norm2, norm3])

        # Final projection to vocabulary
        self.output_proj = layers.Dense(
            self.task_config.vocab_size, name=f"{self.name}_output_proj"
        )

    def call(
        self,
        inputs: Dict[str, keras.KerasTensor],
        training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        vision_features = inputs["vision_features"]
        text_features = inputs["text_features"]  # Assumes pre-embedded text

        x = text_features
        seq_len = ops.shape(x)[1]
        causal_mask = keras.ops.triu(ops.ones((seq_len, seq_len), dtype="bool"), k=1)

        for i in range(self.num_layers):
            # Self-attention with causal mask
            attn_output = self.self_attention_layers[i](
                x, attention_mask=causal_mask, training=training
            )
            x = self.norm_layers[i * 3](x + attn_output)

            # Cross-attention to vision features
            cross_attn_output = self.cross_attention_layers[i](
                x, context=vision_features, training=training
            )
            x = self.norm_layers[i * 3 + 1](x + cross_attn_output)

            # FFN
            ffn_output = self.ffn_layers[i](x, training=training)
            x = self.norm_layers[i * 3 + 2](x + ffn_output)

        logits = self.output_proj(x)
        return {"logits": logits, "hidden_states": x}

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "task_config": self.task_config.__dict__,
                "vision_dim": self.vision_dim,
                "text_dim": self.text_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "ffn_type": self.ffn_type,
            }
        )
        return config


# ---------------------------------------------------------------------
# Visual Question Answering Head
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class VQAHead(keras.layers.Layer):
    """
    A multimodal fusion and classification head for Visual Question Answering.

    This layer is designed to solve the VQA task by first fusing representations
    from vision and text modalities into a single, joint vector, and then
    passing this joint representation through a classifier to predict the final answer.
    It supports several fusion strategies like mean, max, and attention pooling.

    Args:
        task_config (VLMTaskConfig): Configuration object for the task.
        vision_dim (int): Dimension of vision features.
        text_dim (int): Dimension of text features.
        hidden_dims (List[int]): List of hidden layer dimensions for the classifier MLP.
        pooling_strategy (str): Strategy for pooling features ("mean", "max", "attention").
        **kwargs: Additional arguments for the base Layer.
    """

    def __init__(
        self,
        task_config: VLMTaskConfig,
        vision_dim: int = 768,
        text_dim: int = 768,
        hidden_dims: List[int] = [512, 256],
        pooling_strategy: str = "attention",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=f"{task_config.name}_head", **kwargs)

        if task_config.num_classes is None or task_config.num_classes <= 0:
            raise ValueError("VQAHead requires a positive num_classes in task_config.")
        if pooling_strategy not in ["mean", "max", "attention"]:
            raise ValueError(f"Unsupported pooling_strategy: {pooling_strategy}")

        # Store configuration
        self.task_config = task_config
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.hidden_dims = hidden_dims
        self.pooling_strategy = pooling_strategy
        self.embed_dim = self.task_config.hidden_size or max(vision_dim, text_dim)

        # CREATE sub-layers
        if self.pooling_strategy == "attention":
            self.attention_pooling = create_attention_layer(
                "multi_head",
                dim=self.embed_dim,
                num_heads=8,
                dropout_rate=self.task_config.dropout_rate,
                name="attention_pooling",
            )
        else:
            self.attention_pooling = None

        self.hidden_layers = []
        self.dropout_layers = []
        for i, hidden_dim in enumerate(self.hidden_dims):
            self.hidden_layers.append(
                layers.Dense(hidden_dim, activation="gelu", name=f"hidden_{i}")
            )
            self.dropout_layers.append(
                layers.Dropout(self.task_config.dropout_rate, name=f"dropout_{i}")
            )

        self.output_layer = layers.Dense(
            self.task_config.num_classes, name="output_layer"
        )

    def call(
        self,
        inputs: Dict[str, keras.KerasTensor],
        training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        vision_features = inputs["vision_features"]
        question_features = inputs["question_features"]

        if self.pooling_strategy == "mean":
            vision_pooled = ops.mean(vision_features, axis=1)
            text_pooled = ops.mean(question_features, axis=1)
        elif self.pooling_strategy == "max":
            vision_pooled = ops.max(vision_features, axis=1)
            text_pooled = ops.max(question_features, axis=1)
        elif self.pooling_strategy == "attention":
            vision_attended = self.attention_pooling(
                vision_features, context=question_features, training=training
            )
            text_attended = self.attention_pooling(
                question_features, context=vision_features, training=training
            )
            vision_pooled = ops.mean(vision_attended, axis=1)
            text_pooled = ops.mean(text_attended, axis=1)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        x = ops.concatenate([vision_pooled, text_pooled], axis=-1)

        for hidden_layer, dropout_layer in zip(self.hidden_layers, self.dropout_layers):
            x = hidden_layer(x)
            x = dropout_layer(x, training=training)

        logits = self.output_layer(x)
        return {"answer_logits": logits}

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "task_config": self.task_config.__dict__,
                "vision_dim": self.vision_dim,
                "text_dim": self.text_dim,
                "hidden_dims": self.hidden_dims,
                "pooling_strategy": self.pooling_strategy,
            }
        )
        return config


# ---------------------------------------------------------------------
# Visual Grounding Head
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class VisualGroundingHead(BaseVLMHead):
    """Head for visual grounding tasks."""

    def __init__(self, **kwargs: Any) -> None:
        # A strategy that scores per-region interactions is best.
        kwargs.setdefault("fusion_strategy", "gated")
        super().__init__(**kwargs)

        self.bbox_regressor = layers.Dense(4, name=f"{self.name}_bbox_regressor")
        self.confidence_scorer = layers.Dense(
            1, activation="sigmoid", name=f"{self.name}_confidence"
        )

    def call(
        self,
        inputs: Dict[str, keras.KerasTensor],
        training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        vision_features = inputs["vision_features"]  # [B, N_regions, D_vis]
        text_features = inputs["text_features"]

        if len(ops.shape(vision_features)) != 3:
            raise ValueError("VisualGrounding requires spatial vision features.")

        # Pool text features to a single query vector.
        text_query = (
            ops.mean(text_features, axis=1)
            if len(ops.shape(text_features)) == 3
            else text_features
        )

        # Align text query with each visual region for fusion.
        num_regions = ops.shape(vision_features)[1]
        text_expanded = ops.expand_dims(text_query, axis=1)
        text_expanded = ops.tile(text_expanded, [1, num_regions, 1])

        # Fuse each region with the text query. The output is [B, N_regions, D_fused].
        fused_per_region = self.fusion([vision_features, text_expanded], training=training)

        # Score each aligned region's features.
        region_scores = self.confidence_scorer(fused_per_region)
        region_scores = ops.squeeze(region_scores, axis=-1)  # [B, N_regions]

        # Regress bounding box from the top-scoring region's features.
        top_indices = ops.argmax(region_scores, axis=1)
        batch_indices = ops.arange(ops.shape(vision_features)[0])
        top_features = fused_per_region[batch_indices, top_indices]
        bbox = self.bbox_regressor(top_features)

        return {
            "bbox": ops.sigmoid(bbox),
            "confidence": region_scores,
            "grounded_features": top_features,
        }


# ---------------------------------------------------------------------
# Image-Text Matching Head
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ImageTextMatchingHead(BaseVLMHead):
    """
    A projection head for contrastive image-text alignment and fine-grained matching.

    This head performs two functions:
    1.  **Contrastive Alignment**: Projects vision and text features into a
        shared, L2-normalized embedding space for CLIP-style contrastive loss,
        scaled by a learnable temperature.
    2.  **Fine-grained Matching**: Fuses the features using a dedicated fusion
        module to produce a single matching score (0 to 1), indicating the
        semantic correspondence between a specific image-text pair.

    Args:
        task_config (VLMTaskConfig): Configuration object for the task.
        vision_dim (int): Dimension of vision features.
        text_dim (int): Dimension of text features.
        projection_dim (int): Projection dimension for contrastive learning.
        temperature (float): Initial temperature for contrastive loss.
        **kwargs: Additional arguments for the base Layer.
    """

    def __init__(
        self,
        task_config: VLMTaskConfig,
        vision_dim: int = 768,
        text_dim: int = 768,
        projection_dim: int = 256,
        temperature: float = 0.07,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("fusion_strategy", "concatenation")
        super().__init__(
            task_config=task_config, vision_dim=vision_dim, text_dim=text_dim, **kwargs
        )

        self.projection_dim = projection_dim
        self.vision_proj = layers.Dense(projection_dim, name=f"{self.name}_vision_proj")
        self.text_proj = layers.Dense(projection_dim, name=f"{self.name}_text_proj")
        self.similarity_head = layers.Dense(
            1, activation="sigmoid", name=f"{self.name}_similarity"
        )
        self.temperature = self.add_weight(
            name=f"{self.name}_temperature",
            shape=(),
            initializer=keras.initializers.Constant(temperature),
            trainable=True,
        )

    def call(
        self,
        inputs: Dict[str, keras.KerasTensor],
        training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        vision_features = inputs["vision_features"]
        text_features = inputs["text_features"]

        vision_pooled = (
            ops.mean(vision_features, axis=1)
            if len(ops.shape(vision_features)) == 3
            else vision_features
        )
        text_pooled = (
            ops.mean(text_features, axis=1)
            if len(ops.shape(text_features)) == 3
            else text_features
        )

        # 1. Contrastive Alignment part
        vision_projected = self.vision_proj(vision_pooled)
        text_projected = self.text_proj(text_pooled)
        vision_norm = ops.l2_normalize(vision_projected, axis=-1)
        text_norm = ops.l2_normalize(text_projected, axis=-1)
        similarity_matrix = ops.matmul(vision_norm, ops.transpose(text_norm))
        logits = similarity_matrix / self.temperature

        # 2. Fine-grained Matching Score part
        fused = self.fusion([vision_pooled, text_pooled], training=training)
        processed = self.post_fusion_norm(fused, training=training)
        if self.use_post_fusion_ffn:
            processed = self.post_fusion_ffn(processed, training=training)
        match_score = self.similarity_head(processed)

        return {
            "similarity_matrix": similarity_matrix,
            "logits": logits,
            "match_score": ops.squeeze(match_score, axis=-1),
            "vision_embeddings": vision_norm,
            "text_embeddings": text_norm,
        }

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "projection_dim": self.projection_dim,
                # temperature is a weight, will be saved automatically
            }
        )
        return config


# ---------------------------------------------------------------------
# Multi-Task VLM Head
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class MultiTaskVLMHead(keras.layers.Layer):
    """Multi-task head combining multiple VLM task-specific heads."""

    def __init__(
        self,
        task_configs: Dict[str, VLMTaskConfig],
        shared_vision_dim: int = 768,
        shared_text_dim: int = 768,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.task_configs = task_configs
        self.shared_vision_dim = shared_vision_dim
        self.shared_text_dim = shared_text_dim
        self.shared_head_kwargs = kwargs

        self.task_heads = {}
        for task_name, task_config in task_configs.items():
            head_class = get_head_class(task_config.task_type)

            # Combine shared kwargs with task-specific overrides.
            head_kwargs = self.shared_head_kwargs.copy()
            task_specific_kwargs = self.shared_head_kwargs.get(
                "task_specific_kwargs", {}
            ).get(task_name, {})
            head_kwargs.update(task_specific_kwargs)
            if "task_specific_kwargs" in head_kwargs:
                del head_kwargs["task_specific_kwargs"]

            head = head_class(
                task_config=task_config,
                vision_dim=self.shared_vision_dim,
                text_dim=self.shared_text_dim,
                **head_kwargs,
            )
            self.task_heads[task_name] = head
            setattr(self, f"head_{task_name}", head)

    def call(
        self,
        inputs: Dict[str, keras.KerasTensor],
        task_name: Optional[str] = None,
        training: Optional[bool] = None,
    ) -> Union[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        if task_name:
            if task_name not in self.task_heads:
                raise ValueError(f"Unknown task: {task_name}")
            return self.task_heads[task_name](inputs, training=training)

        outputs = {}
        for name, head in self.task_heads.items():
            outputs[name] = head(inputs, training=training)
        return outputs

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "task_configs": {
                    name: tc.__dict__ for name, tc in self.task_configs.items()
                },
                "shared_vision_dim": self.shared_vision_dim,
                "shared_text_dim": self.shared_text_dim,
            }
        )
        config.update(self.shared_head_kwargs)
        return config


# ---------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------


def get_head_class(task_type: VLMTaskType) -> type:
    """Gets the appropriate head class for a VLM task type."""
    head_mapping = {
        VLMTaskType.IMAGE_CAPTIONING: ImageCaptioningHead,
        VLMTaskType.DENSE_CAPTIONING: BaseVLMHead,  # Placeholder
        VLMTaskType.VISUAL_QUESTION_ANSWERING: VQAHead,
        VLMTaskType.VISUAL_GROUNDING: VisualGroundingHead,
        VLMTaskType.IMAGE_TEXT_MATCHING: ImageTextMatchingHead,
        VLMTaskType.VISUAL_DIALOGUE: BaseVLMHead,  # Placeholder
    }
    return head_mapping.get(task_type, BaseVLMHead)


def create_vlm_head(
    task_config: Union[VLMTaskConfig, Dict[str, Any]], **kwargs: Any
) -> Union[BaseVLMHead, keras.layers.Layer]:
    """
    Factory function to create VLM task heads.

    Args:
        task_config: VLMTaskConfig object or dict with task configuration.
        **kwargs: Additional configuration parameters for the head, including
                  `vision_dim`, `text_dim`, `fusion_strategy`, etc.

    Returns:
        A configured VLM head for the specified task.
    """
    if isinstance(task_config, dict):
        task_config = VLMTaskConfig(**task_config)

    head_class = get_head_class(task_config.task_type)
    return head_class(task_config=task_config, **kwargs)


def create_multi_task_vlm_head(
    task_configs: Union[List[VLMTaskConfig], Dict[str, VLMTaskConfig]],
    **kwargs: Any,
) -> MultiTaskVLMHead:
    """
    Creates a multi-task VLM head from task configurations.

    Args:
        task_configs: List or dict of VLMTaskConfig objects.
        **kwargs: Shared configuration for all heads, such as `vision_dim`,
                  `text_dim`, `fusion_strategy`. Can also include
                  `task_specific_kwargs` to override settings for specific tasks.

    Returns:
        MultiTaskVLMHead instance.
    """
    if isinstance(task_configs, list):
        task_configs = {config.name: config for config in task_configs}

    return MultiTaskVLMHead(task_configs=task_configs, **kwargs)