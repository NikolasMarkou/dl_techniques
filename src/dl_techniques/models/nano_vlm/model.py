"""
NanoVLM: Compact Vision-Language Model - Modern Implementation

## Architecture Overview

The NanoVLM combines three main components through a modern, configurable design:

```
Input Images (batch, height, width, channels) + Text Tokens (batch, seq_len)
           ↓                                            ↓
    VisionEncoder                                TextDecoder/TextEncoder
    (configurable)                               (configurable)
           ↓                                            ↓
    Vision Features                              Text Features
    (batch, vision_seq_len, embed_dim)          (batch, text_seq_len, embed_dim)
           ↓                                            ↓
                        MultiModalFusion
                     (configurable strategy)
                              ↓
                    Fused Representations
                   (batch, combined_seq_len, embed_dim)
                              ↓
                       Output Projection
                              ↓
                    Language Model Logits
                   (batch, seq_len, vocab_size)
```

## Key Improvements Over Original

1. **Component Reuse**: Leverages existing VisionEncoder, TextDecoder, and MultiModalFusion
2. **Modern Patterns**: Follows latest Keras 3 serialization and build patterns
3. **Flexible Fusion**: Supports multiple fusion strategies (cross-attention, concatenation, etc.)
4. **Better Abstraction**: Clear separation of concerns between components
5. **Enhanced Configuration**: Comprehensive configuration management and validation
6. **Robust Serialization**: Full save/load compatibility with proper weight restoration

## Usage Examples

### Basic Configuration
```python
# Create model with standard components
model = NanoVLM(
    vision_config={
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12
    },
    text_config={
        'vocab_size': 32000,
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'max_seq_len': 512
    },
    fusion_config={
        'fusion_strategy': 'cross_attention',
        'num_fusion_layers': 6
    }
)
```

### Advanced Configuration
```python
# Modern encoder with advanced components
model = NanoVLM(
    vision_config={
        'img_size': 384,
        'embed_dim': 1024,
        'depth': 24,
        'attention_type': 'differential_attention',
        'normalization_type': 'rms_norm',
        'ffn_type': 'swiglu'
    },
    text_config={
        'vocab_size': 50000,
        'embed_dim': 1024,
        'embedding_type': 'factorized',
        'positional_type': 'rope',
        'normalization_type': 'rms_norm'
    },
    fusion_config={
        'fusion_strategy': 'tensor_fusion',
        'num_tensor_projections': 8
    },
    vocab_size=50000
)
```
"""

import keras
from keras import ops, layers, initializers, regularizers
from typing import Dict, Optional, Tuple, Union, Any, Literal

# ---------------------------------------------------------------------
# Local imports - leveraging existing dl-techniques components
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.text_decoder import TextDecoder
from dl_techniques.layers.text_encoder import TextEncoder
from dl_techniques.layers.multimodal_fusion import MultiModalFusion
from dl_techniques.layers.vision_encoder import VisionEncoder, create_vision_encoder

# ---------------------------------------------------------------------
# Type definitions for enhanced type safety
# ---------------------------------------------------------------------

TextComponentType = Literal['decoder', 'encoder']
FusionStrategy = Literal[
    'cross_attention', 'concatenation', 'addition', 'multiplication',
    'gated', 'attention_pooling', 'bilinear', 'tensor_fusion'
]

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class NanoVLM(keras.Model):
    """
    NanoVLM: Modern Compact Vision-Language Model using existing dl-techniques components.

    A completely rewritten vision-language model that follows modern Keras 3 patterns
    and leverages existing components from the dl-techniques framework. This model
    combines configurable vision encoding, flexible multi-modal fusion, and robust
    text processing through a unified, serializable architecture.

    **Intent**: Provide a production-ready, configurable vision-language model that
    demonstrates proper integration of existing framework components while following
    modern Keras 3 design patterns for robust serialization and deployment.

    **Architecture Components**:
    1. **VisionEncoder**: Configurable vision transformer with multiple architectural options
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
        where combined_sequence_length includes both vision and text tokens.

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

        # Enhance vision config for sequence output
        if self.vision_config.get('output_mode') != 'none':
            logger.info("Setting vision encoder output_mode to 'none' for sequence features")
            self.vision_config['output_mode'] = 'none'

        # Enhance fusion config with embedding dimension
        self.fusion_config['embed_dim'] = vision_dim

        # Set fusion strategy-specific defaults
        fusion_strategy = self.fusion_config['fusion_strategy']
        if fusion_strategy == 'cross_attention' and 'num_heads' not in self.fusion_config:
            self.fusion_config['num_heads'] = self.vision_config['num_heads']

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

        # Build vision encoder
        self.vision_encoder.build(image_shape)
        logger.debug(f"Built vision encoder with input shape: {image_shape}")

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

        # Handle weight sharing if enabled
        if (self.use_shared_embedding and
            self.text_component_type == 'decoder' and
            hasattr(self.text_component, 'word_embeddings')):
            # Tie weights between input embedding and output projection
            self.output_projection.kernel = self.text_component.word_embeddings.embeddings

        # Always call parent build at the end
        super().build(input_shape)
        logger.info("NanoVLM build completed successfully")

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

        # 1. Process images through vision encoder
        vision_features = self.vision_encoder(images, training=training)
        logger.debug(f"Vision features shape: {ops.shape(vision_features)}")

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
        logger.debug(f"Text features shape: {ops.shape(text_features)}")

        # 3. Fuse modalities using MultiModalFusion
        fused_features = self.fusion_layer(
            [vision_features, text_features], training=training
        )
        logger.debug(f"Fused features shape: {ops.shape(fused_features) if not isinstance(fused_features, tuple) else [ops.shape(f) for f in fused_features]}")

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

        logits = self.output_projection(combined_features)
        logger.debug(f"Output logits shape: {ops.shape(logits)}")

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
        """
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

            # Fuse modalities
            fused = self.fusion_layer([vision_features, text_features], training=False)
            if isinstance(fused, tuple):
                vision_fused, text_fused = fused
                combined = ops.concatenate([vision_fused, text_fused], axis=1)
            else:
                combined = fused

            # Get logits and sample next token
            logits = self.output_projection(combined)

            # Extract text logits (skip vision tokens)
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

        # Compute vision sequence length
        vision_output_shape = self.vision_encoder.compute_output_shape(
            input_shape['images'] if isinstance(input_shape, dict) else input_shape[0]
        )
        vision_seq_len = vision_output_shape[1]

        # Combined sequence length (strategy dependent)
        if self.fusion_config['fusion_strategy'] == 'attention_pooling':
            # Pooling strategies return fixed size
            combined_seq_len = 1
        else:
            # Most strategies preserve or combine sequences
            combined_seq_len = vision_seq_len + (text_seq_len or 512)

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
        mini_model = create_nanovlm('mini', fusion_strategy='concatenation')
        base_model = create_nanovlm('base', fusion_strategy='cross_attention')
        large_model = create_nanovlm('large', fusion_strategy='tensor_fusion')
        ```
    """
    variants = {
        'mini': {
            'vision_config': {
                'img_size': 224, 'patch_size': 16, 'embed_dim': 384,
                'depth': 6, 'num_heads': 6, 'output_mode': 'none'
            },
            'text_config': {
                'vocab_size': vocab_size, 'embed_dim': 384, 'depth': 6,
                'num_heads': 6, 'max_seq_len': 512
            },
            'fusion_config': {
                'fusion_strategy': fusion_strategy, 'embed_dim': 384,
                'num_heads': 6, 'num_fusion_layers': 3
            }
        },
        'base': {
            'vision_config': {
                'img_size': 224, 'patch_size': 16, 'embed_dim': 768,
                'depth': 12, 'num_heads': 12, 'output_mode': 'none'
            },
            'text_config': {
                'vocab_size': vocab_size, 'embed_dim': 768, 'depth': 12,
                'num_heads': 12, 'max_seq_len': 512
            },
            'fusion_config': {
                'fusion_strategy': fusion_strategy, 'embed_dim': 768,
                'num_heads': 12, 'num_fusion_layers': 6
            }
        },
        'large': {
            'vision_config': {
                'img_size': 384, 'patch_size': 16, 'embed_dim': 1024,
                'depth': 24, 'num_heads': 16, 'output_mode': 'none'
            },
            'text_config': {
                'vocab_size': vocab_size, 'embed_dim': 1024, 'depth': 24,
                'num_heads': 16, 'max_seq_len': 1024
            },
            'fusion_config': {
                'fusion_strategy': fusion_strategy, 'embed_dim': 1024,
                'num_heads': 16, 'num_fusion_layers': 8
            }
        }
    }

    if variant not in variants:
        raise ValueError(f"Unknown variant '{variant}'. Available: {list(variants.keys())}")

    config = variants[variant]

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

    Example:
        ```python
        model = create_modern_nanovlm(
            vocab_size=50000,
            embed_dim=1024,
            fusion_strategy='tensor_fusion'
        )
        ```
    """
    return NanoVLM(
        vision_config={
            'img_size': 224, 'patch_size': 16, 'embed_dim': embed_dim,
            'depth': 12, 'num_heads': embed_dim // 64,
            'attention_type': 'differential_attention',
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
            'fusion_strategy': kwargs.get('fusion_strategy', 'tensor_fusion'),
            'embed_dim': embed_dim,
            'num_heads': embed_dim // 64,
            'num_tensor_projections': 8,
            'ffn_type': 'swiglu',
            'norm_type': 'rms_norm'
        },
        vocab_size=vocab_size,
        **{k: v for k, v in kwargs.items() if k != 'fusion_strategy'}
    )

# ---------------------------------------------------------------------