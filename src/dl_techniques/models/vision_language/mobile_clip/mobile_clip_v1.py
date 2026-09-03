"""MobileCLIP v1: a faithful CLIP text tower paired with a substitute image tower.

Builds `MobileClipModel`, a two-tower contrastive model: an image encoder
and a text encoder each map their input to one L2-normalized embedding
space, scored by a learned temperature. The official image backbones
(`mci0`, `mci1`, `mci2`, `vit_b16`) have no `keras.applications` equivalent,
so `components._BACKBONE_ALIASES` resolves each name to a real CNN instead
(decision D-001) — variant `b` runs a MobileNetV3Large, not a ViT-B/16. The
text tower is a plain, faithful CLIP transformer, shared with v2. Causal
masking is per-variant: `b` uses it, `s0`/`s1`/`s2` do not.

No pretrained weights ship with this package, and no MobileCLIP number
should be compared against this model's output: `create_mobile_clip_model(pretrained=True)`
raises `NotImplementedError` rather than silently returning random weights.
Warm-start from a local checkpoint with `model.load_weights(path)` instead.
The faithful FastViT-based port is `mobile_clip_v2.py`.

References:
    - Vasu et al., 2023. MobileCLIP: Fast Image-Text Models through Multi-Modal
      Reinforced Training. (https://arxiv.org/abs/2311.17049)
    - Radford et al., 2021. Learning Transferable Visual Models From Natural
      Language Supervision. (https://arxiv.org/abs/2103.00020)
    - Howard et al., 2019. Searching for MobileNetV3.
      (https://arxiv.org/abs/1905.02244)
    - Sandler et al., 2018. MobileNetV2: Inverted Residuals and Linear
      Bottlenecks. (https://arxiv.org/abs/1801.04381)
"""

import math
import keras
from keras import ops
from typing import Optional, Union, Tuple, Dict, Any, List

from dl_techniques.utils.logger import logger
from .components import MobileClipTextEncoder, MobileClipImageEncoder
from dl_techniques.utils.keras_registration import register_dl_technique


@register_dl_technique("dl_techniques.models.mobile_clip.mobile_clip_v1")
class MobileClipModel(keras.Model):
    """MobileCLIP dual encoder: image and text towers into one embedding space.

    Architecture:

    .. code-block:: text

        image [B,H,W,3]              text [B,L]
              │                          │
              ▼                          ▼
        image_encoder              text_encoder
              │                          │
              ▼                          ▼
        L2 normalize                L2 normalize
              │                          │
              ▼                          ▼
        image_features             text_features
              (scaled by logit_scale, clipped to [0, 100])

    Variants: ``b`` (ViT-B16-named, resolved to a CNN, 12-layer text),
    ``s0`` (MCI0-named, 4-layer text), ``s1``/``s2`` (MCI1/MCI2-named,
    12-layer text). See the module docstring for the image-backbone caveat.

    :param embed_dim: Shared embedding dimension for both modalities. Must be positive.
    :type embed_dim: int
    :param image_config: Image encoder configuration, including `backbone_name`, `image_size`.
    :type image_config: Dict[str, Any]
    :param text_config: Text encoder configuration, including `vocab_size`, `max_seq_len`.
    :type text_config: Dict[str, Any]
    :param logit_scale_init: Initial value for the learnable logit scale. Defaults to ``ln(1/0.07)``.
    :type logit_scale_init: float
    :param output_dict: Whether to return outputs as a dict rather than a tuple. Defaults to `True`.
    :type output_dict: bool
    :param kwargs: Additional `keras.Model` arguments.

    :ivar image_encoder: The `MobileClipImageEncoder` instance.
    :ivar text_encoder: The `MobileClipTextEncoder` instance.
    :ivar logit_scale: Learnable temperature parameter for similarity scaling.

    Input shape:
        Dict with keys ``'image'`` (``(B, H, W, 3)``) and ``'text'`` (``(B, L)``).

    Output shape:
        If `output_dict`, a dict with keys `'image_features'`, `'text_features'`,
        `'logit_scale'`; otherwise the tuple `(image_features, text_features, logit_scale)`.

    Example:
        >>> model = MobileClipModel.from_variant('s0')
        >>> inputs = {
        ...     'image': keras.random.normal((32, 224, 224, 3)),
        ...     'text': keras.random.randint(0, 49408, (32, 77)),
        ... }
        >>> outputs = model(inputs)
    """

    MODEL_VARIANTS = {
        "b": {
            "embed_dim": 512,
            "image_config": {
                "backbone_name": "vit_b16",
                "image_size": 224,
                "backbone_weights": None,
                "backbone_trainable": True,
                "projection_dropout_rate": 0.1,
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
                "backbone_weights": None,
                "backbone_trainable": True,
                "projection_dropout_rate": 0.1,
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
                "backbone_weights": None,
                "backbone_trainable": True,
                "projection_dropout_rate": 0.1,
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
                "backbone_weights": None,
                "backbone_trainable": True,
                "projection_dropout_rate": 0.1,
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
        super().build(input_shape)

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


def create_mobile_clip_model(
        variant: str = "s0",
        pretrained: bool = False,
        **kwargs: Any
) -> MobileClipModel:
    """Build a Mobile CLIP model from a named variant.

    :param variant: One of `MobileClipModel.MODEL_VARIANTS` (default `"s0"`).
    :param pretrained: Must stay `False`; no checkpoints ship with this package.
    :return: The constructed model.
    :raises NotImplementedError: If `pretrained=True`.
    """
    # DECISION plan-2026-08-14T233721-d4f9beb2/D-069: raise, do not warn-and-continue.
    if pretrained:
        raise NotImplementedError(
            f"No pretrained MobileCLIP weights are distributed with dl_techniques "
            f"(requested variant '{variant}'). Build the architecture with "
            f"pretrained=False and warm-start from a local checkpoint instead: "
            f"model = create_mobile_clip_model('{variant}', ...); "
            f"model.load_weights('/path/to/weights.keras'). Prefer "
            f"dl_techniques.utils.weight_transfer.load_weights_or_raise(model, "
            f"path), which raises when a load changes ZERO variables -- raw "
            f"load_weights is silent about a checkpoint that matches nothing."
        )
    model = MobileClipModel.from_variant(variant, **kwargs)
    return model

# ---------------------------------------------------------------------
