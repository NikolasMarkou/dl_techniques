"""
NanoVLM, built by the ``NanoVLM`` class, combines a vision encoder, a text
tower, and a multi-modal fusion layer into a vision-language model with one
shared vocabulary head.

An image is a grid of pixels with no order; a caption is a sequence of
vocabulary indices whose order is the point. This model makes the image look
like text first: a vision transformer emits a sequence of patch tokens the
same width as the text embeddings, and a configurable ``MultiModalFusion``
layer (eight strategies) decides how the two sequences interact, from
cross-attention that keeps both streams intact down to a plain
concatenation. Six of the eight strategies require the vision and text
sequences to be the same length, which this model cannot guarantee, so it
raises a clear error naming the strategy instead of failing inside a
concatenation op.

The vision encoder is always forced to ``output_mode='none'`` so it returns
a sequence rather than a pooled vector. The output is unnormalized logits over
the vocabulary at every position of the fused sequence; no softmax is applied,
so a loss takes ``from_logits=True``. Input/output embedding tying happens at
call time via a matrix multiply against the transposed embedding table, not by
reassigning a layer's weight, which Keras 3 forbids after build; a tied model
builds no output ``Dense`` at all.
``generate()`` only supports the ``'cross_attention'`` fusion strategy — the
other seven either lose the per-token axis or cannot keep the vision and
text streams at matching lengths once generation starts appending tokens.
Generation itself is a plain sampling loop with no key-value cache, so cost
grows with the square of the number of generated tokens; it is meant for
evaluating a trained model, not for serving.

References:
    - Alayrac et al., 2022. Flamingo: a Visual Language Model for Few-Shot
      Learning. (https://arxiv.org/abs/2204.14198)
    - Liu et al., 2023. Visual Instruction Tuning. (https://arxiv.org/abs/2304.08485)
    - Dosovitskiy et al., 2020. An Image is Worth 16x16 Words: Transformers for
      Image Recognition at Scale. (https://arxiv.org/abs/2010.11929)
    - Zadeh et al., 2017. Tensor Fusion Network for Multimodal Sentiment
      Analysis. (https://arxiv.org/abs/1707.07250)
    - Press and Wolf, 2016. Using the Output Embedding to Improve Language
      Models. (https://arxiv.org/abs/1608.05859)
"""

import copy
import keras

from typing import Dict, Optional, Tuple, Union, Any, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.transformers.text_decoder import TextDecoder
from dl_techniques.layers.transformers.text_encoder import TextEncoder
from dl_techniques.layers.fusion.multimodal_fusion import MultiModalFusion, FusionStrategy
from dl_techniques.layers.transformers.vision_encoder import VisionEncoder, create_vision_encoder
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.tied_embeddings import tied_embedding_logits
from dl_techniques.utils.weight_transfer import refuse_pretrained

# ---------------------------------------------------------------------
# Type definitions for enhanced type safety
# ---------------------------------------------------------------------

TextComponentType = Literal['decoder', 'encoder']

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.nano_vlm.model")
class NanoVLM(keras.Model):
    """
    Vision-language model combining a vision encoder, a text tower, and a
    multi-modal fusion layer under one vocabulary head.

    Architecture:

    .. code-block:: text

        images [B,H,W,C]          text_tokens [B,T]
              │                          │
              ▼                          ▼
        ┌────────────────┐        ┌───────────────────┐
        │ VisionEncoder  │        │ TextDecoder /     │
        │ output_mode=   │        │ TextEncoder       │
        │ 'none'         │        │                   │
        └───────┬────────┘        └───────────┬───────┘
                │ [B,Sv,D]                    │ [B,St,D]
                └───────────┬─────────────────┘
                            ▼
                ┌─────────────────────────┐
                │ MultiModalFusion        │  8 strategies
                └───────────┬─────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼ 'cross_attention'         ▼ other 7 strategies
        (vision_fused, text_fused)     single tensor [B,S,D]
              │
        concat on sequence axis
              ▼
        combined [B, Sv+S or S, D]
                            │
                            ▼
              ┌─────────────────────────┐
              │ output_projection or    │  tied to word
              │ tied embedding matmul   │  embeddings if
              └───────────┬─────────────┘  use_shared_embedding
                          ▼
              logits [B, combined_len, vocab_size]

    :param vision_config: Configuration for the vision encoder. Must include
        ``img_size``, ``patch_size``, ``embed_dim``, ``depth``, ``num_heads``.
    :type vision_config: Dict[str, Any]
    :param text_config: Configuration for the text component. Must include
        ``vocab_size``, ``embed_dim``, ``depth``, ``num_heads``, ``max_seq_len``.
    :type text_config: Dict[str, Any]
    :param fusion_config: Configuration for :class:`MultiModalFusion`. Must
        include ``fusion_strategy``.
    :type fusion_config: Dict[str, Any]
    :param vocab_size: Vocabulary size for embeddings and the output
        projection. Must be positive. Defaults to ``32000``.
    :type vocab_size: int
    :param text_component_type: ``'decoder'`` for causal generation or
        ``'encoder'`` for bidirectional encoding. Defaults to ``'decoder'``.
    :type text_component_type: TextComponentType
    :param use_shared_embedding: Tie the output to the decoder's word
        embedding table: the logits are ``features @ embeddings.T`` and no
        output ``Dense`` is built. Only applies when
        ``text_component_type='decoder'``; an encoder model is always untied.
        A decoder with ``embedding_type='factorized'`` has no full table to
        tie to and refuses ``True``. Defaults to ``True``.
    :type use_shared_embedding: bool
    :param output_dropout_rate: Dropout before the vocabulary projection, in
        ``[0, 1]``. Defaults to ``0.1``.
    :type output_dropout_rate: float
    :param initializer_range: Standard deviation of the truncated-normal
        initializer of the untied ``output_projection``; the towers and the
        fusion layer use their own initializers. Must be positive. Defaults
        to ``0.02``.
    :type initializer_range: float
    :param kernel_regularizer: Optional regularizer for the untied
        ``output_projection`` kernel. A tied model has no such kernel.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Forwarded to ``keras.Model``; an unknown keyword raises
        ``ValueError``.

    Input shape:
        Dict with:

        - ``'images'``: ``(batch_size, height, width, channels)``
        - ``'text_tokens'``: ``(batch_size, sequence_length)`` token IDs

        Optional: ``'attention_mask'``, ``'token_type_ids'`` (encoder only).

    Output shape:
        ``(batch_size, combined_sequence_length, vocab_size)``.

    :ivar vision_encoder: Vision encoder producing patch-token sequences.
    :ivar text_component: TextDecoder or TextEncoder producing text features.
    :ivar fusion_layer: MultiModalFusion instance joining both streams.
    :ivar output_projection: Bias-free ``Dense(vocab_size)`` for an untied
        model, ``None`` for a tied one (the logits then come from the word
        embedding table, and no second vocabulary matrix is built).
    :ivar final_dropout: Dropout applied before the vocabulary projection.

    Example:

    .. code-block:: python

        model = NanoVLM(
            vision_config={
                'img_size': 224, 'patch_size': 16, 'embed_dim': 768,
                'depth': 12, 'num_heads': 12, 'output_mode': 'none'
            },
            text_config={
                'vocab_size': 32000, 'embed_dim': 768, 'depth': 12,
                'num_heads': 12, 'max_seq_len': 512
            },
            fusion_config={
                'fusion_strategy': 'cross_attention',
                'num_fusion_layers': 6, 'attention_config': {'num_heads': 12}
            },
            vocab_size=32000
        )
        inputs = {
            'images': keras.random.normal((2, 224, 224, 3)),
            'text_tokens': keras.random.randint((2, 128), 0, 32000)
        }
        logits = model(inputs, training=True)

    :raises ValueError: If a configuration dict is missing a required key,
        if the vision and text embedding dimensions disagree, or if a
        numeric parameter is out of range.
    """

    #: Public-name registry of the four named NanoVLM configurations, read by
    #: :meth:`from_variant`. ``vocab_size`` is a caller argument, not a variant
    #: property, so it is absent here; ``from_variant`` writes it into a deep
    #: copy of the selected entry. ``fusion_strategy`` is each row's default,
    #: which a caller may override.
    #:
    #: ``'modern'`` is the former ``create_modern_nanovlm`` configuration:
    #: RMSNorm, SwiGLU, differential attention in the vision tower, RoPE and
    #: factorized word embeddings in the text tower. A factorized table cannot
    #: be tied to the output (it is ``vocab x 128`` followed by a projection),
    #: so the row is untied and builds its own ``output_projection``.
    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
        'mini': {
            'vision_config': {
                'img_size': 224, 'patch_size': 16, 'embed_dim': 384,
                'depth': 6, 'num_heads': 6, 'output_mode': 'none'
            },
            'text_config': {
                'embed_dim': 384, 'depth': 6,
                'num_heads': 6, 'max_seq_len': 512
            },
            'fusion_config': {
                'fusion_strategy': 'cross_attention', 'dim': 384,
                'attention_config': {'num_heads': 6}, 'num_fusion_layers': 3
            }
        },
        'base': {
            'vision_config': {
                'img_size': 224, 'patch_size': 16, 'embed_dim': 768,
                'depth': 12, 'num_heads': 12, 'output_mode': 'none'
            },
            'text_config': {
                'embed_dim': 768, 'depth': 12,
                'num_heads': 12, 'max_seq_len': 512
            },
            'fusion_config': {
                'fusion_strategy': 'cross_attention', 'dim': 768,
                'attention_config': {'num_heads': 12}, 'num_fusion_layers': 6
            }
        },
        'large': {
            'vision_config': {
                'img_size': 384, 'patch_size': 16, 'embed_dim': 1024,
                'depth': 24, 'num_heads': 16, 'output_mode': 'none'
            },
            'text_config': {
                'embed_dim': 1024, 'depth': 24,
                'num_heads': 16, 'max_seq_len': 1024
            },
            'fusion_config': {
                'fusion_strategy': 'cross_attention', 'dim': 1024,
                'attention_config': {'num_heads': 16}, 'num_fusion_layers': 8
            }
        },
        'modern': {
            'vision_config': {
                'img_size': 224, 'patch_size': 16, 'embed_dim': 768,
                'depth': 12, 'num_heads': 12,
                'attention_type': 'differential',
                'normalization_type': 'rms_norm',
                'normalization_position': 'pre',
                'ffn_type': 'swiglu',
                'output_mode': 'none'
            },
            'text_config': {
                'embed_dim': 768, 'depth': 12,
                'num_heads': 12, 'max_seq_len': 1024,
                'embedding_type': 'factorized',
                'positional_type': 'rope',
                'normalization_type': 'rms_norm',
                'normalization_position': 'pre',
                'ffn_type': 'swiglu'
            },
            'fusion_config': {
                # DECISION plan-2026-08-14T183218-f4c612aa/D-007: default stays
                # 'cross_attention', not 'tensor_fusion' -- this factory's fixed 197-token vision stream violates tensor_fusion's equal-length need on every call. See decisions.md.
                'fusion_strategy': 'cross_attention',
                'dim': 768,
                'attention_config': {'num_heads': 12},
                'num_tensor_projections': 8,
                'ffn_type': 'swiglu',
                'norm_type': 'rms_norm'
            },
            'use_shared_embedding': False
        }
    }

    def __init__(
            self,
            vision_config: Dict[str, Any],
            text_config: Dict[str, Any],
            fusion_config: Dict[str, Any],
            vocab_size: int = 32000,
            text_component_type: TextComponentType = 'decoder',
            use_shared_embedding: bool = True,
            output_dropout_rate: float = 0.1,
            initializer_range: float = 0.02,
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate basic parameters
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if text_component_type not in ('decoder', 'encoder'):
            raise ValueError(
                f"text_component_type must be 'decoder' or 'encoder', got "
                f"{text_component_type!r}"
            )
        if not (0.0 <= output_dropout_rate <= 1.0):
            raise ValueError(f"output_dropout_rate must be between 0.0 and 1.0, got {output_dropout_rate}")
        if initializer_range <= 0.0:
            raise ValueError(f"initializer_range must be positive, got {initializer_range}")

        # Store ALL configuration parameters for serialization (CRITICAL for Keras 3)
        self.vision_config = copy.deepcopy(vision_config)
        self.text_config = copy.deepcopy(text_config)
        self.fusion_config = copy.deepcopy(fusion_config)
        self.vocab_size = vocab_size
        self.text_component_type = text_component_type
        self.use_shared_embedding = use_shared_embedding
        self.output_dropout_rate = output_dropout_rate
        self.initializer_range = initializer_range
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # Validate and enhance configurations
        self._validate_and_prepare_configs()

        # The ONE place that decides where the logits come from; build(),
        # call() and generate() all read it.
        self.ties_output_to_embeddings = (
            use_shared_embedding and text_component_type == 'decoder'
        )

        # CREATE all sub-layers in __init__ (Modern Keras 3 Pattern)
        self.vision_encoder = self._create_vision_encoder()
        self.text_component = self._create_text_component()
        self.fusion_layer = self._create_fusion_layer()

        if self.ties_output_to_embeddings and not hasattr(self.text_component, 'word_embeddings'):
            raise ValueError(
                "use_shared_embedding=True ties the output to the decoder's "
                "word embedding table, but text_config embedding_type="
                f"{self.text_config.get('embedding_type')!r} builds no such "
                "table. Pass use_shared_embedding=False for a factorized "
                "embedding."
            )

        self.final_dropout = keras.layers.Dropout(
            rate=output_dropout_rate,
            name='final_dropout'
        ) if output_dropout_rate > 0.0 else None

        # DECISION plan-2026-10-14T062616-81938e81/D-037: a tied model builds NO
        # output Dense. Do not bring back a `shared_output_projection` "for
        # serialization" or "for the untied path": it was created, built, saved
        # and counted (24,576,000 params on 'base') and never called, the one
        # trainable variable with a None gradient. The untied path builds its
        # own `output_projection` below. See decisions.md D-037.
        self.output_projection = None if self.ties_output_to_embeddings else keras.layers.Dense(
            self.vocab_size,
            use_bias=False,
            kernel_initializer=keras.initializers.TruncatedNormal(stddev=self.initializer_range),
            kernel_regularizer=self.kernel_regularizer,
            name='output_projection'
        )

    def _validate_and_prepare_configs(self) -> None:
        """Validate and enhance configuration dictionaries with cross-component consistency."""

        # Validate required keys
        required_vision_keys = ['img_size', 'patch_size', 'embed_dim', 'depth', 'num_heads']
        for key in required_vision_keys:
            if key not in self.vision_config:
                raise ValueError(f"Missing required vision_config key: {key}")

        required_text_keys = ['vocab_size', 'embed_dim', 'depth', 'num_heads']
        for key in required_text_keys:
            if key not in self.text_config:
                raise ValueError(f"Missing required text_config key: {key}")

        required_fusion_keys = ['fusion_strategy']
        for key in required_fusion_keys:
            if key not in self.fusion_config:
                raise ValueError(f"Missing required fusion_config key: {key}")

        # Validate embedding dimension consistency
        vision_dim = self.vision_config['embed_dim']
        text_dim = self.text_config['embed_dim']
        if vision_dim != text_dim:
            raise ValueError(
                f"Vision embed_dim ({vision_dim}) must match text embed_dim ({text_dim}) "
                f"for fusion compatibility"
            )

        # Ensure vocab_size consistency
        if self.text_config['vocab_size'] != self.vocab_size:
            logger.warning(
                f"Text config vocab_size ({self.text_config['vocab_size']}) differs from "
                f"model vocab_size ({self.vocab_size}). Using model vocab_size."
            )
            self.text_config['vocab_size'] = self.vocab_size

        # Enhance vision_heads config for sequence output
        if self.vision_config.get('output_mode') != 'none':
            logger.info("Setting vision_heads encoder output_mode to 'none' for sequence features")
            self.vision_config['output_mode'] = 'none'

        # Enhance fusion config with embedding dimension
        self.fusion_config['dim'] = vision_dim

        # Set fusion strategy-specific defaults
        fusion_strategy = self.fusion_config['fusion_strategy']
        if fusion_strategy == 'cross_attention':
            attention_config = self.fusion_config.setdefault('attention_config', {})
            if 'num_heads' not in attention_config:
                attention_config['num_heads'] = self.vision_config['num_heads']

    def _create_vision_encoder(self) -> VisionEncoder:
        """Create the vision tower, named so its weight paths are stable."""
        return create_vision_encoder(**self.vision_config, name='vision_encoder')

    def _create_text_component(self) -> Union[TextDecoder, TextEncoder]:
        """Create text processing component based on type."""
        if self.text_component_type == 'decoder':
            return TextDecoder(**self.text_config, name='text_decoder')
        return TextEncoder(**self.text_config, name='text_encoder')

    def _create_fusion_layer(self) -> MultiModalFusion:
        """Create MultiModalFusion using existing component."""
        # DECISION plan_2026-06-15_39a31d4a/D-002: MultiModalFusion takes `dim` +
        # `attention_config={'num_heads': N}`, not `embed_dim`/`num_heads`.
        # A stale key is forwarded by the **splat and raises. See decisions.md.
        return MultiModalFusion(**self.fusion_config, name='multimodal_fusion')

    def build(self, input_shape: Union[Dict[str, Tuple], Tuple]) -> None:
        """
        Build the NanoVLM and all its sub-layers.

        CRITICAL: Following modern Keras 3 patterns, explicitly build each sub-layer
        to ensure all weight variables exist before weight restoration during loading.
        """
        if self.built:
            return

        # Parse input shapes
        if isinstance(input_shape, dict):
            image_shape = input_shape['images']
            text_shape = input_shape['text_tokens']
        elif isinstance(input_shape, (list, tuple)) and len(input_shape) == 2:
            image_shape, text_shape = input_shape
        else:
            raise ValueError(
                "input_shape must be dict with 'images' and 'text_tokens' keys "
                "or tuple of (image_shape, text_shape)"
            )

        # Build vision_heads encoder
        self.vision_encoder.build(image_shape)
        logger.debug(f"Built vision_heads encoder with input shape: {image_shape}")

        # Build text component
        self.text_component.build(text_shape)
        logger.debug(f"Built text component with input shape: {text_shape}")

        # Compute feature shapes for fusion layer building
        # Vision features: [batch, vision_seq_len, embed_dim]
        vision_seq_len = self.vision_encoder.compute_output_shape(image_shape)[1]
        vision_feature_shape = (None, vision_seq_len, self.vision_config['embed_dim'])

        # Text features: [batch, text_seq_len, embed_dim]
        text_seq_len = text_shape[1] if text_shape[1] is not None else 512
        text_feature_shape = (None, text_seq_len, self.text_config['embed_dim'])

        # Build fusion layer with both modality shapes
        fusion_input_shapes = [vision_feature_shape, text_feature_shape]
        self.fusion_layer.build(fusion_input_shapes)
        logger.debug(f"Built fusion layer with shapes: {fusion_input_shapes}")

        # Compute fusion output shape for final layers
        fusion_output_shape = self.fusion_layer.compute_output_shape(fusion_input_shapes)

        # Handle different fusion strategies output shapes
        if isinstance(fusion_output_shape, tuple) and len(fusion_output_shape) == 2:
            # Cross-attention returns tuple of outputs
            combined_shape = (None, vision_seq_len + text_seq_len, self.vision_config['embed_dim'])
        elif isinstance(fusion_output_shape, (list, tuple)) and len(fusion_output_shape) == 3:
            # Single tensor output
            combined_shape = fusion_output_shape
        else:
            # Fallback shape computation
            combined_shape = (None, vision_seq_len + text_seq_len, self.vision_config['embed_dim'])

        # Build final layers
        if self.final_dropout is not None:
            self.final_dropout.build(combined_shape)

        if self.output_projection is not None:
            self.output_projection.build(combined_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            inputs: Union[Dict[str, keras.KerasTensor], Tuple[keras.KerasTensor, keras.KerasTensor]],
            training: Optional[bool] = None,
            **kwargs
    ) -> keras.KerasTensor:
        """
        Run the forward pass.

        :param inputs: Dict with ``'images'`` and ``'text_tokens'``, or a tuple
            ``(images, text_tokens)``. Optional keys: ``'attention_mask'``,
            ``'token_type_ids'``.
        :type inputs: Union[Dict[str, keras.KerasTensor], Tuple[keras.KerasTensor, keras.KerasTensor]]
        :param training: Whether the model is in training mode.
        :type training: Optional[bool]
        :return: Logits of shape ``(batch, combined_seq_len, vocab_size)``.
        :rtype: keras.KerasTensor
        """
        # Parse inputs
        if isinstance(inputs, dict):
            images = inputs['images']
            text_tokens = inputs['text_tokens']
            attention_mask = inputs.get('attention_mask')
            token_type_ids = inputs.get('token_type_ids')
        else:
            images, text_tokens = inputs
            attention_mask = None
            token_type_ids = None

        # DECISION plan-2026-08-19T163559-499b6f0e/D-084: no logging in call().
        # A `logger.debug` shape line under tf.function logs once at trace time only. See decisions.md.
        vision_features = self.vision_encoder(images, training=training)

        # 2. Process text through text component
        if self.text_component_type == 'decoder':
            text_features = self.text_component(
                text_tokens, attention_mask=attention_mask, training=training
            )
        else:  # encoder
            text_features = self.text_component(
                inputs={'input_ids': text_tokens, 'attention_mask': attention_mask,
                       'token_type_ids': token_type_ids},
                training=training
            )
        # 3. Fuse modalities using MultiModalFusion
        fused_features = self.fusion_layer(
            [vision_features, text_features], training=training
        )
        # 4. Handle fusion strategy outputs
        if isinstance(fused_features, tuple):
            # Cross-attention returns separate outputs - concatenate them
            vision_fused, text_fused = fused_features
            combined_features = keras.ops.concatenate([vision_fused, text_fused], axis=1)
        else:
            # Single tensor output from other strategies
            combined_features = fused_features

        # 5. Apply final dropout and the vocabulary projection
        if self.final_dropout is not None:
            combined_features = self.final_dropout(combined_features, training=training)
        return self._vocab_logits(combined_features)

    def _vocab_logits(self, features: keras.KerasTensor) -> keras.KerasTensor:
        """
        Project features to vocabulary logits, tied or untied.

        Shared by :meth:`call` and :meth:`generate`, so the two cannot disagree
        about where the logits come from.

        :param features: ``(batch, seq, embed_dim)``.
        :type features: keras.KerasTensor
        :return: Unnormalized logits ``(batch, seq, vocab_size)``; no softmax.
        :rtype: keras.KerasTensor
        """
        # DECISION plan_2026-06-15_2a23a001/D-001: tie embeddings at call time via
        # matmul against the transposed embedding table, never by reassigning a built layer's weight. See decisions.md.
        # DECISION plan-2026-09-12T123331-28fd855f/D-005: the matmul/transpose
        # EXPRESSION is consolidated into the shared `tied_embedding_logits`
        # helper (see decisions.md D-005 task 6); this does NOT contradict the
        # D-001 anchor above -- the tie is still resolved at call time via a
        # matmul against the transposed table, never by reassigning a built
        # layer's weight. See decisions.md D-005.
        if self.ties_output_to_embeddings:
            return tied_embedding_logits(
                features,
                self.text_component.word_embeddings.embeddings,
            )
        return self.output_projection(features)

    def generate(
            self,
            images: keras.KerasTensor,
            prompt_tokens: keras.KerasTensor,
            max_length: int = 100,
            temperature: float = 1.0,
            top_k: int = 50,
            eos_token_id: int = 2,
            **kwargs
    ) -> keras.KerasTensor:
        """
        Generate text tokens autoregressively from images and a prompt.

        :param images: Input images, shape ``(batch_size, height, width, channels)``.
        :type images: keras.KerasTensor
        :param prompt_tokens: Initial prompt tokens, shape ``(batch_size, prompt_length)``.
        :type prompt_tokens: keras.KerasTensor
        :param max_length: Maximum number of tokens to generate.
        :type max_length: int
        :param temperature: Sampling temperature.
        :type temperature: float
        :param top_k: Number of highest-probability tokens to sample from.
        :type top_k: int
        :param eos_token_id: Token ID that ends generation.
        :type eos_token_id: int
        :param kwargs: Unused, reserved for future generation parameters.
        :return: Generated token sequence, shape ``(batch_size, total_length)``.
        :rtype: keras.KerasTensor
        :raises ValueError: If the fusion strategy is not ``'cross_attention'``.
        """
        # DECISION plan-2026-08-14T233721-d4f9beb2/D-025: generate() supports only
        # 'cross_attention'; the other 7 strategies break mid-loop (rank-2 output, or unequal stream lengths once tokens append). See decisions.md.
        strategy = self.fusion_config['fusion_strategy']
        if strategy != 'cross_attention':
            raise ValueError(
                f"generate() requires fusion_strategy='cross_attention', got "
                f"'{strategy}'. Only cross_attention keeps the vision and text "
                "streams separate, which is what lets this loop take the last "
                "TEXT position's logits and append a token. "
                + (
                    "'attention_pooling' pools the sequence away entirely "
                    "(output is rank 2), so no per-token logit row exists."
                    if strategy == 'attention_pooling' else
                    f"'{strategy}' needs both streams at the same sequence "
                    "length, which autoregression breaks the moment it appends "
                    "a token to a fixed-length vision stream."
                )
            )

        # Process images once (cached for generation)
        vision_features = self.vision_encoder(images, training=False)

        # Initialize with prompt
        current_tokens = prompt_tokens

        for step in range(max_length):
            # Get current text features
            if self.text_component_type == 'decoder':
                text_features = self.text_component(current_tokens, training=False)
            else:
                text_features = self.text_component(
                    {'input_ids': current_tokens}, training=False
                )

            # Fuse modalities. Only 'cross_attention' reaches this loop (see the
            # D-025 refusal at the top of the method), so the fusion layer always
            # returns the two streams separately here.
            vision_fused, text_fused = self.fusion_layer(
                [vision_features, text_features], training=False
            )
            combined = keras.ops.concatenate([vision_fused, text_fused], axis=1)

            # Get logits and sample next token
            logits = self._vocab_logits(combined)

            # Extract text logits (skip the vision prefix)
            vision_seq_len = keras.ops.shape(vision_features)[1]
            text_logits = logits[:, vision_seq_len:, :]
            next_token_logits = text_logits[:, -1, :]  # Last text position

            # Sample next tokens for all sequences in batch
            next_tokens = self._sample_tokens_batch(
                next_token_logits, temperature, top_k
            )

            # Append to sequences
            next_tokens = keras.ops.expand_dims(next_tokens, axis=1)
            current_tokens = keras.ops.concatenate([current_tokens, next_tokens], axis=1)

            # Check for EOS (simplified - could be enhanced for per-sequence)
            if eos_token_id in next_tokens:
                break

        return current_tokens

    def _sample_tokens_batch(
            self,
            logits: keras.KerasTensor,
            temperature: float,
            top_k: int
    ) -> keras.KerasTensor:
        """Sample next tokens for a batch of sequences."""
        if temperature != 1.0:
            logits = logits / temperature

        if top_k > 0:
            # Top-k sampling
            top_k_logits, top_k_indices = keras.ops.top_k(logits, k=top_k)
            # `categorical` takes LOGITS. Feeding it softmax probabilities
            # samples from softmax(softmax(z)), a flattened distribution.
            sampled_indices = keras.random.categorical(top_k_logits, num_samples=1)[:, 0]

            # Map back to original vocabulary
            next_tokens = keras.ops.take_along_axis(
                top_k_indices, keras.ops.expand_dims(sampled_indices, axis=1), axis=1
            )[:, 0]
        else:
            # Greedy sampling
            next_tokens = keras.ops.argmax(logits, axis=-1)

        return next_tokens

    def compute_output_shape(self, input_shape: Union[Dict, Tuple]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape for a given input shape.

        :param input_shape: Dict with ``'images'``/``'text_tokens'`` shapes, or
            a tuple of ``(image_shape, text_shape)``.
        :type input_shape: Union[Dict, Tuple]
        :return: Output shape ``(batch_size, combined_seq_len, vocab_size)``.
        :rtype: Tuple[Optional[int], ...]
        """
        if isinstance(input_shape, dict):
            batch_size = input_shape['images'][0]
            text_seq_len = input_shape['text_tokens'][1]
        else:
            batch_size = input_shape[0][0]
            text_seq_len = input_shape[1][1]

        # Compute vision_heads sequence length
        vision_output_shape = self.vision_encoder.compute_output_shape(
            input_shape['images'] if isinstance(input_shape, dict) else input_shape[0]
        )
        vision_seq_len = vision_output_shape[1]

        # DECISION plan-2026-08-14T233721-d4f9beb2/D-026: ask the fusion layer for
        # the fused length rather than re-deriving vision_seq_len + text_seq_len — only 'cross_attention' actually sums. See decisions.md.
        text_feature_dim = self.fusion_config.get('dim', self.text_config['embed_dim'])
        fused_shape = self.fusion_layer.compute_output_shape([
            (batch_size, vision_seq_len, vision_output_shape[-1]),
            (batch_size, text_seq_len, text_feature_dim),
        ])

        if self.fusion_config['fusion_strategy'] == 'cross_attention':
            # A per-modality shape list; `call` concatenates them on axis 1.
            vision_part, text_part = fused_shape
            if vision_part[1] is None or text_part[1] is None:
                combined_seq_len = None
            else:
                combined_seq_len = vision_part[1] + text_part[1]
        elif len(fused_shape) == 2:
            # Pooled to (batch, dim): the logits lose the sequence axis too.
            return (batch_size, self.vocab_size)
        else:
            combined_seq_len = fused_shape[1]

        return (batch_size, combined_seq_len, self.vocab_size)

    @classmethod
    def from_variant(
            cls,
            variant: str,
            vocab_size: int = 32000,
            fusion_strategy: Optional[FusionStrategy] = None,
            text_component_type: TextComponentType = 'decoder',
            pretrained: Union[bool, str] = False,
            **kwargs: Any
    ) -> "NanoVLM":
        """
        Build a NanoVLM from a row of :attr:`MODEL_VARIANTS`.

        Only ``'cross_attention'`` and ``'attention_pooling'`` can fuse streams
        of different sequence length. The other six strategies combine on the
        feature axis or broadcast element-wise on the sequence axis, so they
        need matching vision and text lengths. Every variant fixes the vision
        length from ``img_size``/``patch_size`` (197 tokens for ``'mini'``,
        ``'base'`` and ``'modern'``, 577 for ``'large'``), so those six raise a
        ``ValueError`` on the first call unless the caller's text length
        happens to match; the text length is not known until a batch arrives,
        so this cannot be checked at construction. All eight strategies can be
        trained at matched lengths, but ``generate()`` accepts only
        ``'cross_attention'``.

        :param variant: One of ``'mini'``, ``'base'``, ``'large'``, ``'modern'``.
        :type variant: str
        :param vocab_size: Vocabulary size, written into the text config.
        :type vocab_size: int
        :param fusion_strategy: Overrides the row's ``fusion_strategy``
            (``'cross_attention'`` in every row). ``None`` keeps the row's.
        :type fusion_strategy: Optional[FusionStrategy]
        :param text_component_type: ``'decoder'`` or ``'encoder'``.
        :type text_component_type: TextComponentType
        :param pretrained: Must be falsy. No NanoVLM weights are distributed,
            and the model has no backbone/head split to load partially: its
            output is the vocabulary logits of the whole model, tied to the
            word embedding table. ``True`` or a path raises
            ``NotImplementedError`` before anything is built; load a local
            archive with ``keras.models.load_model(path)``.
        :type pretrained: Union[bool, str]
        :param kwargs: Laid over the row. ``vision_config``, ``text_config``
            and ``fusion_config`` are MERGED one level deep into the row's
            dicts (a key you pass replaces that key; the others stay); every
            other keyword replaces the row's value or goes to the constructor.
            An unknown keyword raises ``ValueError``.
        :type kwargs: Any
        :return: A configured, unbuilt model.
        :rtype: NanoVLM
        :raises ValueError: If ``variant`` is not a known row.
        :raises NotImplementedError: If ``pretrained`` is truthy.

        Example:

        .. code-block:: python

            model = NanoVLM.from_variant('mini', vocab_size=1000)
            small = NanoVLM.from_variant(
                'modern', vocab_size=64,
                vision_config={'embed_dim': 64, 'num_heads': 1},
                text_config={'embed_dim': 64, 'num_heads': 1},
                fusion_config={'attention_config': {'num_heads': 1}},
            )
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Available: {list(cls.MODEL_VARIANTS.keys())}"
            )
        refuse_pretrained(pretrained, "NanoVLM")

        # Deep-copied: the caller's values are written into the copy, and
        # writing them into the class table itself would make every later call
        # inherit this call's values.
        config = copy.deepcopy(cls.MODEL_VARIANTS[variant])
        for key in ('vision_config', 'text_config', 'fusion_config'):
            config[key].update(copy.deepcopy(kwargs.pop(key, {})))
        config.update(kwargs)
        config['text_config']['vocab_size'] = vocab_size
        if fusion_strategy is not None:
            config['fusion_config']['fusion_strategy'] = fusion_strategy

        return cls(
            vocab_size=vocab_size,
            text_component_type=text_component_type,
            **config
        )

    def get_config(self) -> Dict[str, Any]:
        """
        Get the model configuration for serialization.

        :return: Constructor arguments needed to reconstruct this model.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'vision_config': self.vision_config,
            'text_config': self.text_config,
            'fusion_config': self.fusion_config,
            'vocab_size': self.vocab_size,
            'text_component_type': self.text_component_type,
            'use_shared_embedding': self.use_shared_embedding,
            'output_dropout_rate': self.output_dropout_rate,
            'initializer_range': self.initializer_range,
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------

def create_nanovlm(
        variant: str = "base",
        pretrained: Union[bool, str] = False,
        **kwargs: Any
) -> NanoVLM:
    """
    Build a NanoVLM from a named variant; delegates to :meth:`NanoVLM.from_variant`.

    :param variant: One of ``'mini'``, ``'base'``, ``'large'``, ``'modern'``.
    :type variant: str
    :param pretrained: Must be falsy; see :meth:`NanoVLM.from_variant`.
    :type pretrained: Union[bool, str]
    :param kwargs: Forwarded to :meth:`NanoVLM.from_variant`.
    :return: A configured model.
    :rtype: NanoVLM
    :raises ValueError: If ``variant`` is not a known preset.
    :raises NotImplementedError: If ``pretrained`` is truthy.

    Example:

    .. code-block:: python

        model = create_nanovlm('base', vocab_size=32000)
    """
    return NanoVLM.from_variant(variant, pretrained=pretrained, **kwargs)

# ---------------------------------------------------------------------
