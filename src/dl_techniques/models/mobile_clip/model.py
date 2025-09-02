import math
import keras
from keras import ops
from typing import Optional, Union, Tuple, Dict, Any, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from .components import MobileClipTextEncoder, MobileClipImageEncoder


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MobileClipModel(keras.Model):
    """
    Mobile CLIP Model combining image and text encoders with variant support.

    This model implements the complete Mobile CLIP architecture, combining
    separate image and text encoders to produce embeddings in a shared
    latent space. It follows modern Keras 3 patterns with comprehensive
    variant handling similar to other dl-techniques models.

    **Architecture**:
    ```
    Image Input                    Text Input
         ↓                             ↓
    MobileClipImageEncoder      MobileClipTextEncoder
         ↓                             ↓
    Image Embedding              Text Embedding
         ↓                             ↓
    L2 Normalization            L2 Normalization
         ↓                             ↓
         └─────── Similarity ──────────┘
                (scaled by logit_scale)
    ```

    Model Variants:
    --------------
    - MobileClip-B: Base variant with ViT-B16 image encoder and 12-layer text encoder
    - MobileClip-S0: Compact variant with MCI0 image encoder and 4-layer text encoder
    - MobileClip-S1: Small variant with MCI1 image encoder and 12-layer text encoder
    - MobileClip-S2: Small variant with MCI2 image encoder and 12-layer text encoder

    Args:
        embed_dim: Integer, shared embedding dimension for both modalities.
            Must be positive.
        image_config: Dictionary containing image encoder configuration.
            Should include 'backbone_name', 'image_size', etc.
        text_config: Dictionary containing text encoder configuration.
            Should include 'vocab_size', 'max_seq_len', etc.
        logit_scale_init: Float, initial value for the learnable logit scale.
            Defaults to ln(1/0.07) ≈ 2.66.
        output_dict: Boolean, whether to return outputs as dictionary.
            Defaults to True.
        **kwargs: Additional arguments for the Model base class.

    Input shape:
        Dictionary with keys:
        - 'image': 4D tensor `(batch_size, height, width, 3)`
        - 'text': 2D tensor `(batch_size, sequence_length)`

    Output shape:
        If output_dict=True: Dictionary with keys 'image_features',
        'text_features', 'logit_scale'.
        If output_dict=False: Tuple (image_features, text_features, logit_scale).

    Attributes:
        image_encoder: MobileClipImageEncoder instance.
        text_encoder: MobileClipTextEncoder instance.
        logit_scale: Learnable temperature parameter for similarity scaling.

    Example:
        ```python
        # Create from variant
        model = MobileClipModel.from_variant('s0')

        # Create custom model
        image_config = {
            'backbone_name': 'vit_b16',
            'image_size': 224,
            'backbone_trainable': True,
            'projection_dropout': 0.1
        }

        text_config = {
            'vocab_size': 49408,
            'max_seq_len': 77,
            'embed_dim': 512,
            'num_layers': 12,
            'num_heads': 8,
            'intermediate_size': 2048,
            'use_causal_mask': True
        }

        model = MobileClipModel(
            embed_dim=512,
            image_config=image_config,
            text_config=text_config
        )

        # Use model
        inputs = {
            'image': keras.random.normal((32, 224, 224, 3)),
            'text': keras.random.randint(0, 49408, (32, 77))
        }

        outputs = model(inputs)
        ```

    Note:
        The logit_scale parameter is learned during training and controls
        the temperature for contrastive learning. It's initialized to
        ln(1/0.07) following the CLIP paper.
    """

    # Model variant configurations based on official Mobile CLIP variants
    MODEL_VARIANTS = {
        "b": {
            "embed_dim": 512,
            "image_config": {
                "backbone_name": "vit_b16",
                "image_size": 224,
                "backbone_trainable": True,
                "projection_dropout": 0.1,
            },
            "text_config": {
                "vocab_size": 49408,
                "max_seq_len": 77,  # context_length
                "embed_dim": 512,  # dim from JSON
                "num_layers": 12,  # n_transformer_layers
                "num_heads": 8,  # n_heads_per_layer
                "intermediate_size": 2048,  # dim * ffn_multiplier_per_layer
                "dropout_rate": 0.1,
                "attention_dropout_rate": 0.1,
                "use_causal_mask": True,  # causal_masking
                "model_name": "base",
            }
        },
        "s0": {
            "embed_dim": 512,
            "image_config": {
                "backbone_name": "mci0",
                "image_size": 256,
                "backbone_trainable": True,
                "projection_dropout": 0.1,
            },
            "text_config": {
                "vocab_size": 49408,
                "max_seq_len": 77,
                "embed_dim": 512,
                "num_layers": 4,  # n_transformer_layers (reduced for S0)
                "num_heads": 8,
                "intermediate_size": 2048,
                "dropout_rate": 0.1,
                "attention_dropout_rate": 0.1,
                "use_causal_mask": False,  # causal_masking is false for S variants
                "model_name": "mct",
            }
        },
        "s1": {
            "embed_dim": 512,
            "image_config": {
                "backbone_name": "mci1",
                "image_size": 256,
                "backbone_trainable": True,
                "projection_dropout": 0.1,
            },
            "text_config": {
                "vocab_size": 49408,
                "max_seq_len": 77,
                "embed_dim": 512,
                "num_layers": 12,
                "num_heads": 8,
                "intermediate_size": 2048,
                "dropout_rate": 0.1,
                "attention_dropout_rate": 0.1,
                "use_causal_mask": False,
                "model_name": "base",
            }
        },
        "s2": {
            "embed_dim": 512,
            "image_config": {
                "backbone_name": "mci2",
                "image_size": 256,
                "backbone_trainable": True,
                "projection_dropout": 0.1,
            },
            "text_config": {
                "vocab_size": 49408,
                "max_seq_len": 77,
                "embed_dim": 512,
                "num_layers": 12,
                "num_heads": 8,
                "intermediate_size": 2048,
                "dropout_rate": 0.1,
                "attention_dropout_rate": 0.1,
                "use_causal_mask": False,
                "model_name": "base",
            }
        }
    }

    def __init__(
            self,
            embed_dim: int,
            image_config: Dict[str, Any],
            text_config: Dict[str, Any],
            logit_scale_init: float = math.log(1.0 / 0.07),
            output_dict: bool = True,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if not isinstance(image_config, dict):
            raise TypeError("image_config must be a dictionary")
        if not isinstance(text_config, dict):
            raise TypeError("text_config must be a dictionary")

        self.embed_dim = embed_dim
        self.image_config = image_config.copy()
        self.text_config = text_config.copy()
        self.logit_scale_init = logit_scale_init
        self.output_dict = output_dict

        image_constructor_config = self.image_config.copy()
        text_constructor_config = self.text_config.copy()

        text_constructor_config.pop('model_name', None)

        image_constructor_config['projection_dim'] = embed_dim
        text_constructor_config['projection_dim'] = embed_dim

        self.image_encoder = MobileClipImageEncoder(**image_constructor_config, name='image_encoder')
        self.text_encoder = MobileClipTextEncoder(**text_constructor_config, name='text_encoder')

        self.logit_scale = self.add_weight(
            name='logit_scale',
            shape=(),
            initializer=keras.initializers.Constant(self.logit_scale_init),
            trainable=True,
        )

    def build(self, input_shape: Dict[str, Union[Tuple[int, ...], List[int]]]) -> None:
        """Build the model and its sub-components."""
        if "image" in input_shape and hasattr(self.image_encoder, 'build'):
            self.image_encoder.build(input_shape["image"])
        if "text" in input_shape and hasattr(self.text_encoder, 'build'):
            self.text_encoder.build(input_shape["text"])
        self.built = True

    def encode_image(
            self,
            image: keras.KerasTensor,
            normalize: bool = True,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Encode images to embedding vectors.
        """
        features = self.image_encoder(image, training=training)
        if normalize:
            features = ops.normalize(features, axis=-1)
        return features

    def encode_text(
            self,
            text: keras.KerasTensor,
            normalize: bool = True,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Encode text tokens to embedding vectors.
        """
        features = self.text_encoder(text, training=training)
        if normalize:
            features = ops.normalize(features, axis=-1)
        return features

    def call(
            self,
            inputs: Dict[str, keras.KerasTensor],
            training: Optional[bool] = None
    ) -> Union[Dict[str, keras.KerasTensor], Tuple[keras.KerasTensor, ...]]:
        """
        Forward pass for the MobileClip model.
        """
        image = inputs.get('image')
        text = inputs.get('text')

        image_features = self.encode_image(image, normalize=True, training=training) if image is not None else None
        text_features = self.encode_text(text, normalize=True, training=training) if text is not None else None

        logit_scale = ops.exp(self.logit_scale)
        logit_scale = ops.clip(logit_scale, 0.0, 100.0)

        if self.output_dict:
            return {
                'image_features': image_features,
                'text_features': text_features,
                'logit_scale': logit_scale,
            }
        else:
            return image_features, text_features, logit_scale

    @classmethod
    def from_variant(
            cls,
            variant: str,
            **kwargs: Any
    ) -> "MobileClipModel":
        """
        Create a Mobile CLIP model from a predefined variant.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant].copy()
        logger.info(f"Creating Mobile CLIP-{variant.upper()} model")
        config.update(kwargs)
        return cls(**config)

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'image_config': self.image_config,
            'text_config': self.text_config,
            'logit_scale_init': self.logit_scale_init,
            'output_dict': self.output_dict,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MobileClipModel":
        """Create model from configuration."""
        return cls(**config)

    def summary(self, **kwargs):
        """Print model summary with additional information."""
        super().summary(**kwargs)
        logger.info(f"Mobile CLIP configuration:")
        logger.info(f"  - Embed dimension: {self.embed_dim}")
        logger.info(f"  - Image backbone: {self.image_config.get('backbone_name', 'Unknown')}")
        logger.info(f"  - Image size: {self.image_config.get('image_size', 'Unknown')}")
        logger.info(f"  - Text vocab size: {self.text_config.get('vocab_size', 'Unknown')}")
        logger.info(f"  - Text max seq len: {self.text_config.get('max_seq_len', 'Unknown')}")
        logger.info(f"  - Text layers: {self.text_config.get('num_layers', 'Unknown')}")
        logger.info(f"  - Text heads: {self.text_config.get('num_heads', 'Unknown')}")
        logger.info(f"  - Causal masking: {self.text_config.get('use_causal_mask', 'Unknown')}")
        logger.info(f"  - Output format: {'Dictionary' if self.output_dict else 'Tuple'}")


# ---------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------

def create_mobile_clip_model(
        variant: str = "s0",
        pretrained: bool = False,
        **kwargs: Any
) -> MobileClipModel:
    """
    Convenience function to create Mobile CLIP models.
    """
    if pretrained:
        logger.warning("Pretrained weights are not yet implemented")
    model = MobileClipModel.from_variant(variant, **kwargs)
    return model


def create_mobile_clip_base(**kwargs: Any) -> MobileClipModel:
    """Create a Mobile CLIP base model with ViT-B16 encoder."""
    return create_mobile_clip_model("b", **kwargs)


def create_mobile_clip_s0(**kwargs: Any) -> MobileClipModel:
    """Create a Mobile CLIP S0 model (most compact)."""
    return create_mobile_clip_model("s0", **kwargs)


def create_mobile_clip_s1(**kwargs: Any) -> MobileClipModel:
    """Create a Mobile CLIP S1 model."""
    return create_mobile_clip_model("s1", **kwargs)


def create_mobile_clip_s2(**kwargs: Any) -> MobileClipModel:
    """Create a Mobile CLIP S2 model."""
    return create_mobile_clip_model("s2", **kwargs)