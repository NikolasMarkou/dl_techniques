"""
MobileCLIP v1 dual encoder — a faithful CLIP text tower paired with a
deliberately substituted image tower.

MobileCLIP's contribution is efficiency rather than a new training objective.
The contrastive premise is CLIP's: two towers map an image and a caption into
one L2-normalized embedding space, and the only supervision is which pairing in
the batch is correct. What MobileCLIP changes is the cost of the image side —
a hybrid convolution/transformer trunk (MCi) whose training-time branches
reparameterize into a single convolution at inference — and the data side, via
multi-modal reinforced training against a captioner-and-ensemble teacher. Only
the first of those is an architectural matter, and it is exactly the part this
module does not implement.

**This class is deliberately non-faithful on the image side, and the
substitution is total.** The official image backbones (``mci0``, ``mci1``,
``mci2``, ``vit_b16``) have no equivalent in ``keras.applications``: there is no
MCi port and no ViT. Since ``ImageProjectionHead`` opens with
``GlobalAveragePooling2D``, the backbone must emit a 4-D ``[B, H, W, C]`` feature
map, which independently rules out a token-sequence ViT. ``components._BACKBONE_ALIASES``
therefore resolves each of those names to a real ``keras.applications`` CNN, so
that variant ``b`` builds and runs a MobileNetV3Large rather than a ViT-B/16.
This is a deliberate choice of functional buildability over weights fidelity,
recorded as the package's D-001; every tabulated variant also sets
``backbone_weights=None``, so not even the substitute's ImageNet weights are
loaded. Nothing here reproduces published MobileCLIP numbers and nothing should
be compared against them. The faithful port lives beside this file in
``mobile_clip_v2.py``, over the real FastViT MCi tower; the two coexist and
neither deprecates the other.

The text tower, by contrast, is a plain CLIP transformer and is faithful. It is
shared verbatim with v2 rather than copied, which is a maintenance decision with
teeth: it owns one of only two places in the tree that adapt a block-polarity
mask into a keep-polarity one, and a copy would create a third. Token embeddings
are scaled by ``embed_dim ** -0.5`` before positional embeddings are added, the
stack ends in a LayerNorm, and a raw ``(embed_dim, projection_dim)`` weight —
not a ``Dense`` — performs the projection into the shared space.

Causal masking is per-variant, not a constant. The ``b`` variant attends
causally; ``s0``, ``s1`` and ``s2`` do not, matching their official configs. When
it is on, the mask is built from ``MaskFactory.create_causal_mask`` and inverted,
rather than from ``ops.tril``: ``ops.tril`` routes through a ``tf.cond`` that
rejects a Python-bool predicate the moment it is traced, so it works eagerly and
fails on every graph path — ``tf.function``, ``predict``, ``.keras`` save/load,
XLA. The mask this layer needs is the complementary keep polarity, hence the
``logical_not`` and cast.

Pooling is CLIP's end-of-text convention, implemented as ``argmax`` over the raw
token ids. That is correct only because CLIP's BPE vocabulary assigns the EOT
token the numerically largest id in a well-formed sequence; a tokenizer that
breaks that property, or a sequence containing an id above EOT, pools the wrong
position silently. The gather is a one-hot matmul so it stays differentiable and
backend-agnostic.

The temperature is stored as a log and exponentiated on use, with the result
clipped to ``[0, 100]`` — the clip is not cosmetic, since an unbounded
temperature turns a diverging run into ``inf`` logits and a ``nan`` loss with no
other visible symptom. Unlike v2, this class stops there: ``call`` returns the
two feature tensors and the scale, never the similarity matrices, and it emits
``None`` for a missing modality rather than omitting the key, so consumers must
check for ``None`` rather than for key presence.

No pretrained weights are distributed. ``create_mobile_clip_model(pretrained=True)``
raises ``NotImplementedError``. It used to log a warning and return a randomly
initialized model, which made an unavailable checkpoint indistinguishable from a
successful load at the call site; warm-start from a local file with
``model.load_weights(path)`` instead.

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

    .. note::
        The official MobileCLIP image backbones (``vit_b16``, ``mci0``, ``mci1``,
        ``mci2``) do NOT exist in ``keras.applications``. To keep every variant
        buildable and able to run forward inference, these names are resolved to
        real ``keras.applications`` CNN backbones via
        ``components._BACKBONE_ALIASES`` (see decision D-001). This is a
        functional substitute, NOT a weights-faithful ViT/MCi port — variant
        ``b`` runs a MobileNetV3Large CNN rather than a true ViT-B/16.

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
                "backbone_weights": None,  # D-001: CNN substitute built from scratch (no imagenet weights)
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
                "backbone_weights": None,  # D-001: CNN substitute built from scratch (no imagenet weights)
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
                "backbone_weights": None,  # D-001: CNN substitute built from scratch (no imagenet weights)
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
                "backbone_weights": None,  # D-001: CNN substitute built from scratch (no imagenet weights)
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

    :raises NotImplementedError: If ``pretrained=True`` — no MobileCLIP
        checkpoints ship with this package.
    """
    # DECISION plan-2026-08-14T233721-d4f9beb2/D-069: raise, do not warn-and-continue.
    if pretrained:
        raise NotImplementedError(
            f"No pretrained MobileCLIP weights are distributed with dl_techniques "
            f"(requested variant '{variant}'). Build the architecture with "
            f"pretrained=False and warm-start from a local checkpoint instead: "
            f"model = create_mobile_clip_model('{variant}', ...); "
            f"model.load_weights('/path/to/weights.keras')."
        )
    model = MobileClipModel.from_variant(variant, **kwargs)
    return model

# ---------------------------------------------------------------------
