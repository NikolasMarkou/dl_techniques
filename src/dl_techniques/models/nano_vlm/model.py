"""
Compact vision-language model assembled from three interchangeable library
components — a vision encoder, a causal or bidirectional text tower, and a
pluggable multi-modal fusion layer — over a shared vocabulary head.

The problem a VLM has to solve is that its two inputs are not commensurate. An
image arrives as a grid of pixels with no discrete units and no order; a caption
arrives as a sequence of vocabulary indices whose order is the whole point. The
standard resolution is to make the image look like text before the language model
sees it: run a vision transformer, keep its patch tokens rather than a pooled
summary, and hand the language side a sequence of vectors it can treat exactly
like word embeddings. Once both modalities are sequences in a single width, any
mechanism that mixes two sequences becomes a candidate for joining them, and the
design question stops being "how do we represent an image" and becomes "where and
how much do the modalities interact".

This model keeps that question open rather than answering it once. Both towers
are required to emit ``embed_dim``-wide sequences — construction fails if the
vision and text widths disagree, since nothing downstream could reconcile them —
and everything about the interaction is delegated to a configurable
``MultiModalFusion`` with eight strategies. Cross-attention lets each modality
read the other while keeping both streams intact; the pooling and tensor-product
strategies collapse them more aggressively; concatenation barely interacts them
at all. The trade is depth of interaction against parameter count and sequence
length, and it is meant to be swept, not decided in this file. Six of the eight
carry a precondition this model cannot satisfy by construction: everything except
``'cross_attention'`` and ``'attention_pooling'`` combines the streams on the
feature axis or broadcasts them element-wise on the sequence axis, and so needs
the vision and text lengths to be EQUAL. The vision length is pinned by
``img_size``/``patch_size``; the text length is the caller's. Those six now raise
a ``ValueError`` naming the strategy on the first call instead of dying inside a
backend ``ConcatOp``.

Two consequences of that delegation are not obvious from the call graph. First,
the vision encoder is forced to ``output_mode='none'`` during config validation
regardless of what the caller asked for: the default CLS pooling would hand the
fusion layer a rank-2 tensor, and every strategy expects a rank-3 sequence.
Second, the fusion layer's output shape is strategy-dependent, so ``call``
branches on the *type* of what it returns — cross-attention yields a tuple of two
streams, which are concatenated along the sequence axis with vision first, while
the other strategies yield a single tensor that is used as-is. The vocabulary
projection is then applied to the whole combined sequence. **Under
cross-attention only**, the logits tensor therefore contains one row per vision
token as well as one per text token, and those leading rows predict nothing:
``generate`` slices them off, and any loss computed against this model must slice
the same way or it will train the head to predict tokens for image patches. Under
every other sequence-preserving strategy the fused tensor is already one row per
position at the single shared length and there is no vision prefix to slice —
slicing one off left an EMPTY axis, which ``generate`` then indexed. That is why
``generate`` now accepts ``'cross_attention'`` and refuses the other seven by
name: ``'attention_pooling'`` has no per-token row at all (rank 2), and the
remaining six require both streams at the same length, which autoregression
breaks the moment it appends a token to a fixed-length vision stream. The fused
length itself is asked of ``MultiModalFusion.compute_output_shape`` rather than
re-derived here, so ``compute_output_shape`` and ``call`` cannot drift apart.

Note that vision tokens are *not* spliced into the text token sequence in the
LLaVA sense — the text tower runs on text alone and the modalities only meet
downstream, in the fusion layer. Whatever causal masking applies is therefore
entirely the text component's business: ``text_component_type='decoder'`` selects
a causal tower for generation, ``'encoder'`` a bidirectional one, and no mask is
constructed here.

Input/output embedding tying is performed at call time, as a
``matmul(x, transpose(word_embeddings.embeddings))``, and never by reassigning a
weight. The straightforward implementation — pointing the output ``Dense``'s
kernel at the embedding matrix — is doubly wrong under Keras 3: reassigning a
built layer's kernel raises outright, and the embedding matrix is
``(vocab, dim)``, the transpose of the Dense kernel's ``(dim, vocab)``, so the
shapes never matched either. The ``output_projection`` layer is still created and
built, because it is the live path when tying is disabled and because a
half-built layer would break serialization, but it is simply unused when tying is
on.

The fusion configuration has a signature hazard worth stating explicitly.
``MultiModalFusion`` takes ``dim`` plus ``attention_config={'num_heads': N}``; it
does *not* take ``embed_dim`` or a top-level ``num_heads``. Because the config
dict is splatted into the constructor, a stale key is not ignored — it is
forwarded to the base ``Layer`` and raises. Both factories in this module write
the current spelling; the older ``embed_dim``/``num_heads`` form was removed from
``create_modern_nanovlm`` in commit ``fd35976cb``.

Generation is a plain sampling loop with no KV cache: the image is encoded once
and reused, but the text tower re-reads the entire prefix at every step, so cost
grows quadratically in the number of generated tokens. It is adequate for
smoke-testing a trained model and not intended as a serving path.

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
from keras import ops, layers, initializers, regularizers
from typing import Dict, Optional, Tuple, Union, Any, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.transformers.text_decoder import TextDecoder
from dl_techniques.layers.transformers.text_encoder import TextEncoder
from dl_techniques.layers.fusion.multimodal_fusion import MultiModalFusion, FusionStrategy
from dl_techniques.layers.transformers.vision_encoder import VisionEncoder, create_vision_encoder

# ---------------------------------------------------------------------
# Type definitions for enhanced type safety
# ---------------------------------------------------------------------

TextComponentType = Literal['decoder', 'encoder']

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class NanoVLM(keras.Model):
    """
    NanoVLM: Modern Compact Vision-Language Model using existing dl-techniques components.

    A completely rewritten vision_heads-language model that follows modern Keras 3 patterns
    and leverages existing components from the dl-techniques framework. This model
    combines configurable vision_heads encoding, flexible multi-modal fusion, and robust
    text processing through a unified, serializable architecture.

    **Intent**: Provide a production-ready, configurable vision_heads-language model that
    demonstrates proper integration of existing framework components while following
    modern Keras 3 design patterns for robust serialization and deployment.

    **Architecture Components**:
    1. **VisionEncoder**: Configurable vision_heads transformer with multiple architectural options
    2. **TextDecoder/TextEncoder**: Flexible text processing with multiple embedding strategies
    3. **MultiModalFusion**: Advanced cross-modal fusion with 8 different strategies
    4. **Output Projection**: Final vocabulary prediction layer with optional weight tying

    **Modern Keras 3 Patterns**:
    - All sub-layers created in `__init__()` following the "create vs build" principle
    - Explicit sub-layer building in `build()` for robust serialization
    - Complete configuration management with all parameters preserved
    - Proper weight restoration lifecycle support
    - Full type safety with comprehensive validation

    Args:
        vision_config: Dictionary containing configuration for the VisionEncoder.
            Should include keys like 'img_size', 'patch_size', 'embed_dim', 'depth',
            'num_heads', and optionally advanced configuration like 'attention_type',
            'normalization_type', 'ffn_type', etc.
        text_config: Dictionary containing configuration for the text component.
            Should include keys like 'vocab_size', 'embed_dim', 'depth', 'num_heads',
            'max_seq_len', and optionally 'embedding_type', 'positional_type', etc.
        fusion_config: Dictionary containing configuration for MultiModalFusion.
            Should include 'fusion_strategy' and strategy-specific parameters like
            'num_fusion_layers', 'attention_type', 'num_tensor_projections', etc.
        vocab_size: Integer, size of the vocabulary for text embeddings and output
            projection. Must be positive. Defaults to 32000.
        text_component_type: TextComponentType, whether to use 'decoder' for causal
            generation or 'encoder' for bidirectional encoding. Defaults to 'decoder'.
        use_shared_embedding: Boolean, whether to tie input and output embeddings
            for memory efficiency. Only applicable when text_component_type='decoder'.
            Defaults to True.
        output_dropout: Float, dropout rate for the final output projection layer.
            Must be between 0.0 and 1.0. Defaults to 0.1.
        initializer_range: Float, standard deviation for weight initialization.
            Must be positive. Defaults to 0.02.
        kernel_initializer: String or Initializer, kernel weight initializer.
            Defaults to 'glorot_uniform'.
        bias_initializer: String or Initializer, bias weight initializer.
            Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer for kernel weights. Defaults to None.
        bias_regularizer: Optional regularizer for bias weights. Defaults to None.
        **kwargs: Additional keyword arguments for the Model base class.

    Input shape:
        Dictionary with keys:
        - 'images': 4D tensor of shape (batch_size, height, width, channels)
        - 'text_tokens': 2D tensor of shape (batch_size, sequence_length) with token IDs

        Optional keys:
        - 'attention_mask': 2D tensor for padding mask
        - 'token_type_ids': 2D tensor for segment embeddings (encoder only)

    Output shape:
        3D tensor of shape (batch_size, combined_sequence_length, vocab_size)
        where combined_sequence_length includes both vision_heads and text tokens.

    Attributes:
        vision_encoder: VisionEncoder instance for image processing.
        text_component: TextDecoder or TextEncoder instance for text processing.
        fusion_layer: MultiModalFusion instance for cross-modal integration.
        output_projection: Dense layer for vocabulary prediction.
        final_dropout: Dropout layer applied before output projection.

    Example:
        ```python
        # Standard configuration
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
                'num_fusion_layers': 6, 'num_heads': 12
            },
            vocab_size=32000
        )

        # Forward pass
        inputs = {
            'images': keras.ops.random.normal((2, 224, 224, 3)),
            'text_tokens': keras.ops.random.randint(0, 32000, (2, 128))
        }
        logits = model(inputs, training=True)
        print(f"Output shape: {logits.shape}")  # (2, vision_seq_len + text_seq_len, 32000)
        ```

    Raises:
        ValueError: If configuration dictionaries are missing required keys.
        ValueError: If dimension parameters are incompatible between components.
        ValueError: If vocab_size doesn't match between text and model configuration.
        ValueError: If any numeric parameter is outside valid range.

    Note:
        This implementation follows the modern Keras 3 patterns documented in the
        "Complete Guide to Modern Keras 3 Custom Layers and Models" and demonstrates
        proper integration of existing framework components for production deployment.
    """

    #: Public-name registry of the three named nanoVLM sizes (models/CLAUDE.md
    #: Axis 2). Hoisted out of ``create_nanovlm``'s body on 2026-08-19, where it
    #: was a local ``variants`` dict that nothing outside the function could
    #: enumerate.
    #:
    #: Two keys the local table carried are DELIBERATELY absent here, because
    #: they are caller arguments and not properties of the variant:
    #: ``text_config['vocab_size']`` and ``fusion_config['fusion_strategy']``.
    #: ``create_nanovlm`` injects both into a ``copy.deepcopy`` of the entry it
    #: selects -- deep-copied, never mutated in place, so this class attribute
    #: cannot be corrupted by a call (the shared-mutable-default failure mode).
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
                'dim': 384,
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
                'dim': 768,
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
                'dim': 1024,
                'attention_config': {'num_heads': 16}, 'num_fusion_layers': 8
            }
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
            output_dropout: float = 0.1,
            initializer_range: float = 0.02,
            kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            bias_regularizer: Optional[regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate basic parameters
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if not (0.0 <= output_dropout <= 1.0):
            raise ValueError(f"output_dropout must be between 0.0 and 1.0, got {output_dropout}")
        if initializer_range <= 0.0:
            raise ValueError(f"initializer_range must be positive, got {initializer_range}")

        # Store ALL configuration parameters for serialization (CRITICAL for Keras 3)
        self.vision_config = vision_config.copy()
        self.text_config = text_config.copy()
        self.fusion_config = fusion_config.copy()
        self.vocab_size = vocab_size
        self.text_component_type = text_component_type
        self.use_shared_embedding = use_shared_embedding
        self.output_dropout = output_dropout
        self.initializer_range = initializer_range
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        # Validate and enhance configurations
        self._validate_and_prepare_configs()

        # CREATE all sub-layers in __init__ (Modern Keras 3 Pattern)
        logger.info("Creating NanoVLM components using existing dl-techniques layers...")

        # 1. Create VisionEncoder using existing component
        self.vision_encoder = self._create_vision_encoder()

        # 2. Create text component (decoder or encoder)
        self.text_component = self._create_text_component()

        # 3. Create MultiModalFusion using existing component
        self.fusion_layer = self._create_fusion_layer()

        # 4. Create output layers
        self.final_dropout = layers.Dropout(
            rate=output_dropout,
            name='final_dropout'
        ) if output_dropout > 0.0 else None

        self.output_projection = self._create_output_projection()

        logger.info("NanoVLM components created successfully using modern Keras 3 patterns.")

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
        """Create VisionEncoder using existing component."""
        try:
            return create_vision_encoder(**self.vision_config)
        except Exception as e:
            logger.error(f"Failed to create VisionEncoder: {e}")
            # Fallback to direct instantiation
            return VisionEncoder(**self.vision_config, name='vision_encoder')

    def _create_text_component(self) -> Union[TextDecoder, TextEncoder]:
        """Create text processing component based on type."""
        if self.text_component_type == 'decoder':
            return TextDecoder(**self.text_config, name='text_decoder')
        else:  # encoder
            return TextEncoder(**self.text_config, name='text_encoder')

    def _create_fusion_layer(self) -> MultiModalFusion:
        """Create MultiModalFusion using existing component."""
        # DECISION plan_2026-06-15_39a31d4a/D-002: MultiModalFusion takes dim +
        # attention_config={'num_heads':N}, NOT embed_dim/num_heads (caller was coded
        # against a ghost API). Every fusion_config source must use the new keys or the
        # **splat re-injects the ghost kwarg. Correct form: layers/heads/vlm/factory.py:121.
        return MultiModalFusion(**self.fusion_config, name='multimodal_fusion')

    def _create_output_projection(self) -> layers.Layer:
        """Create output projection layer with optional weight sharing."""
        if (self.use_shared_embedding and
            self.text_component_type == 'decoder' and
            hasattr(self.text_component, 'word_embeddings')):

            # Create shared embedding projection (tie weights)
            return layers.Dense(
                self.vocab_size,
                use_bias=False,
                kernel_initializer=initializers.TruncatedNormal(stddev=self.initializer_range),
                kernel_regularizer=self.kernel_regularizer,
                name='shared_output_projection'
            )
        else:
            # Standard output projection
            return layers.Dense(
                self.vocab_size,
                use_bias=False,
                kernel_initializer=initializers.TruncatedNormal(stddev=self.initializer_range),
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name='output_projection'
            )

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

        self.output_projection.build(combined_shape)
        logger.debug(f"Built output projection with shape: {combined_shape}")

        # NOTE (plan_2026-06-15_39a31d4a/D-002 cascade): Keras 3 forbids reassigning a
        # built layer's `.kernel` (raises "cannot add new elements of state ... already
        # built"), and `word_embeddings.embeddings` is (vocab, dim) — the transpose of the
        # Dense kernel (dim, vocab) — so the old tie was both illegal AND shape-wrong. The
        # output_projection keeps its own built kernel; weight-tying is dropped (was
        # dead-on-forward code, never executed). Re-add via a proper tied-Dense layer if
        # memory-sharing is later required.

        logger.info("NanoVLM build completed successfully")

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            inputs: Union[Dict[str, keras.KerasTensor], Tuple[keras.KerasTensor, keras.KerasTensor]],
            training: Optional[bool] = None,
            **kwargs
    ) -> keras.KerasTensor:
        """
        Forward pass through NanoVLM.

        Args:
            inputs: Input dictionary with 'images' and 'text_tokens' keys, or tuple
                of (images, text_tokens). Additional optional keys: 'attention_mask',
                'token_type_ids'.
            training: Boolean indicating training mode.
            **kwargs: Additional keyword arguments.

        Returns:
            Language model logits of shape [batch, combined_seq_len, vocab_size].
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

        # 1. Process images through vision_heads encoder
        # DECISION plan-2026-08-19T163559-499b6f0e/D-084: no logging on the
        # forward path (R-033/R-041). The four `logger.debug` shape lines that
        # stood in this `call` each ran `ops.shape(...)` for the log line only;
        # under `tf.function` they emit a symbolic tensor once at trace time and
        # never again. Print shapes from a test or `model.summary()`, not here.
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
            combined_features = ops.concatenate([vision_fused, text_fused], axis=1)
        else:
            # Single tensor output from other strategies
            combined_features = fused_features

        # 5. Apply final dropout and output projection
        if self.final_dropout is not None:
            combined_features = self.final_dropout(combined_features, training=training)

        # DECISION plan_2026-06-15_2a23a001/D-001: tie input/output embeddings at CALL
        # time via matmul(x, transpose(word_embeddings.embeddings)); NEVER reassign another
        # layer's weight post-build (that broke in plan_2026-06-15_39a31d4a). output_projection
        # stays built (serialization / use_shared_embedding=False) but is unused on this path.
        if (self.use_shared_embedding and
                self.text_component_type == 'decoder' and
                hasattr(self.text_component, 'word_embeddings')):
            logits = ops.matmul(
                combined_features,
                ops.transpose(self.text_component.word_embeddings.embeddings)
            )
        else:
            logits = self.output_projection(combined_features)
        return logits

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
        Generate text autoregressively given images and prompt.

        Args:
            images: Input images tensor of shape [batch_size, height, width, channels]
            prompt_tokens: Initial prompt tokens of shape [batch_size, prompt_length]
            max_length: Maximum number of tokens to generate
            temperature: Sampling temperature for controlling randomness
            top_k: Number of highest probability tokens for sampling
            eos_token_id: Token ID that signals end of sequence
            **kwargs: Additional generation parameters

        Returns:
            Generated token sequence of shape [batch_size, total_length]

        Raises:
            ValueError: If the configured fusion strategy is anything other than
                ``'cross_attention'``. See the D-025 anchor below.
        """
        # DECISION plan-2026-08-14T233721-d4f9beb2/D-025
        # `generate()` supports 'cross_attention' and nothing else, and says so.
        #
        # WHAT NOT TO DO: do NOT re-open this loop to the other seven strategies
        # by "handling" their output shape. The loop is structurally incompatible
        # with them, and each fails differently:
        #   * 'attention_pooling' returns rank 2 — there is no per-text-token
        #     logit row to sample from at all.
        #   * the other six require the vision and text streams to have EQUAL
        #     sequence lengths (they combine on the feature axis or broadcast on
        #     the sequence axis). Autoregression appends one token per step while
        #     the vision length is fixed by img_size/patch_size, so the
        #     precondition can hold for at most ONE step. Measured 2026-08-15:
        #     with the prompt padded to the vision length, step 2 raises
        #     "requires all modality inputs to share the same sequence length".
        # Before this guard the loop ran for every strategy and sliced a
        # vision-length prefix off a tensor that had none, leaving an EMPTY axis
        # that `text_logits[:, -1, :]` then indexed. See decisions.md D-025.
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
        batch_size = ops.shape(current_tokens)[0]

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
            combined = ops.concatenate([vision_fused, text_fused], axis=1)

            # Get logits and sample next token
            # Shared-embedding tie at call time (mirrors call(); see D-001 anchor above).
            if (self.use_shared_embedding and
                    self.text_component_type == 'decoder' and
                    hasattr(self.text_component, 'word_embeddings')):
                logits = ops.matmul(
                    combined,
                    ops.transpose(self.text_component.word_embeddings.embeddings)
                )
            else:
                logits = self.output_projection(combined)

            # Extract text logits (skip the vision prefix)
            vision_seq_len = ops.shape(vision_features)[1]
            text_logits = logits[:, vision_seq_len:, :]
            next_token_logits = text_logits[:, -1, :]  # Last text position

            # Sample next tokens for all sequences in batch
            next_tokens = self._sample_tokens_batch(
                next_token_logits, temperature, top_k
            )

            # Append to sequences
            next_tokens = ops.expand_dims(next_tokens, axis=1)
            current_tokens = ops.concatenate([current_tokens, next_tokens], axis=1)

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
            top_k_logits, top_k_indices = ops.top_k(logits, k=top_k)
            probs = ops.softmax(top_k_logits)
            sampled_indices = keras.random.categorical(probs, num_samples=1)[:, 0]

            # Map back to original vocabulary
            next_tokens = ops.take_along_axis(
                top_k_indices, ops.expand_dims(sampled_indices, axis=1), axis=1
            )[:, 0]
        else:
            # Greedy sampling
            next_tokens = ops.argmax(logits, axis=-1)

        return next_tokens

    def compute_output_shape(self, input_shape: Union[Dict, Tuple]) -> Tuple[Optional[int], ...]:
        """Compute output shape given input shape."""
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

        # DECISION plan-2026-08-14T233721-d4f9beb2/D-026
        # The fused length is asked of the fusion layer; it is not re-derived.
        #
        # WHAT NOT TO DO: do NOT restore `combined_seq_len = vision_seq_len +
        # text_seq_len` for every non-'attention_pooling' strategy. That
        # contradicted `MultiModalFusion.compute_output_shape`, which returns a
        # per-modality tuple for 'cross_attention' and `input_shape[0][1]` — the
        # VISION length alone — for all six sequence-preserving strategies. Only
        # 'cross_attention' sums, and only because this model concatenates the
        # tuple on axis 1 itself; the sum was wrong for the other six and the
        # hard-coded 1 was wrong for 'attention_pooling', which pools to rank 2.
        # See decisions.md D-026.
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

    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration for serialization.

        CRITICAL: Must include ALL constructor parameters for complete reconstruction.
        """
        config = super().get_config()
        config.update({
            'vision_config': self.vision_config,
            'text_config': self.text_config,
            'fusion_config': self.fusion_config,
            'vocab_size': self.vocab_size,
            'text_component_type': self.text_component_type,
            'use_shared_embedding': self.use_shared_embedding,
            'output_dropout': self.output_dropout,
            'initializer_range': self.initializer_range,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        })
        return config


# ---------------------------------------------------------------------
# Factory Functions for Convenient Model Creation
# ---------------------------------------------------------------------

def create_nanovlm(
        variant: str = "base",
        vocab_size: int = 32000,
        fusion_strategy: FusionStrategy = 'cross_attention',
        text_component_type: TextComponentType = 'decoder',
        **kwargs
) -> NanoVLM:
    """
    Factory function to create NanoVLM with predefined configurations.

    Args:
        variant: Model size variant ('mini', 'base', 'large')
        vocab_size: Vocabulary size for text processing
        fusion_strategy: Strategy for multi-modal fusion
        text_component_type: Whether to use 'decoder' or 'encoder'
        **kwargs: Additional model parameters

    Returns:
        Configured NanoVLM instance

    Example:
        ```python
        # Create different variants
        mini_model = create_nanovlm('mini', fusion_strategy='cross_attention')
        base_model = create_nanovlm('base', fusion_strategy='cross_attention')
        large_model = create_nanovlm('large', fusion_strategy='cross_attention')
        ```

        Only `'cross_attention'` and `'attention_pooling'` can fuse streams of
        DIFFERENT sequence length. The other six — `'concatenation'`,
        `'tensor_fusion'`, `'addition'`, `'multiplication'`, `'gated'` and
        `'bilinear'` — combine on the feature axis or broadcast element-wise on
        the sequence axis, so they require the vision and text lengths to be
        equal. Every variant here fixes the vision length from
        `img_size`/`patch_size` (197 tokens for `'mini'`/`'base'`, 577 for
        `'large'`) against a caller-chosen text length, so those six raise a
        `ValueError` naming the strategy and the requirement on the first call
        unless the two happen to match. (Construction cannot decide it: the text
        length is not known until a batch arrives.) All eight strategies can be
        TRAINED at matched lengths, but `generate()` accepts `'cross_attention'`
        alone and refuses the rest by name — see its own docstring.
    """
    if variant not in NanoVLM.MODEL_VARIANTS:
        raise ValueError(
            f"Unknown variant '{variant}'. "
            f"Available: {list(NanoVLM.MODEL_VARIANTS.keys())}"
        )

    # Deep-copied: `vocab_size` and `fusion_strategy` are caller arguments, and
    # writing them into the class table itself would make every later call
    # inherit this call's values.
    config = copy.deepcopy(NanoVLM.MODEL_VARIANTS[variant])
    config['text_config']['vocab_size'] = vocab_size
    config['fusion_config']['fusion_strategy'] = fusion_strategy

    return NanoVLM(
        vision_config=config['vision_config'],
        text_config=config['text_config'],
        fusion_config=config['fusion_config'],
        vocab_size=vocab_size,
        text_component_type=text_component_type,
        **kwargs
    )


def create_modern_nanovlm(
        vocab_size: int = 32000,
        embed_dim: int = 768,
        **kwargs
) -> NanoVLM:
    """
    Create NanoVLM with modern architectural components.

    Uses advanced components like RMSNorm, SwiGLU, differential attention, etc.

    The fusion default is `'cross_attention'`, matching `create_nanovlm`. It used to
    be `'tensor_fusion'`, which could not run on any input this factory accepts:
    `tensor_fusion` concatenates the modalities on the feature axis and therefore
    needs equal sequence lengths, while this factory's vision stream is fixed at 197
    tokens (`img_size=224`, `patch_size=16`) against a caller-chosen text length.
    Passing `fusion_strategy='tensor_fusion'` explicitly is still allowed; it now
    raises a `ValueError` naming the length requirement instead of dying inside a
    `ConcatOp`.

    Example:
        ```python
        model = create_modern_nanovlm(
            vocab_size=50000,
            embed_dim=1024
        )
        ```
    """
    return NanoVLM(
        vision_config={
            'img_size': 224, 'patch_size': 16, 'embed_dim': embed_dim,
            'depth': 12, 'num_heads': embed_dim // 64,
            'attention_type': 'differential',
            'normalization_type': 'rms_norm',
            'normalization_position': 'pre',
            'ffn_type': 'swiglu',
            'output_mode': 'none'
        },
        text_config={
            'vocab_size': vocab_size, 'embed_dim': embed_dim, 'depth': 12,
            'num_heads': embed_dim // 64, 'max_seq_len': 1024,
            'embedding_type': 'factorized',
            'positional_type': 'rope',
            'normalization_type': 'rms_norm',
            'normalization_position': 'pre',
            'ffn_type': 'swiglu'
        },
        fusion_config={
            # DECISION plan-2026-08-14T183218-f4c612aa/D-007
            # Do NOT restore 'tensor_fusion' here for symmetry with the docstring
            # example or with MultiModalFusion's own richer strategy: this factory
            # hardcodes a 197-token vision stream, so tensor_fusion's equal-length
            # requirement is violated by EVERY call, not by an edge case. See
            # decisions.md D-007.
            'fusion_strategy': kwargs.get('fusion_strategy', 'cross_attention'),
            'dim': embed_dim,
            'attention_config': {'num_heads': embed_dim // 64},
            'num_tensor_projections': 8,
            'ffn_type': 'swiglu',
            'norm_type': 'rms_norm'
        },
        vocab_size=vocab_size,
        **{k: v for k, v in kwargs.items() if k != 'fusion_strategy'}
    )

# ---------------------------------------------------------------------