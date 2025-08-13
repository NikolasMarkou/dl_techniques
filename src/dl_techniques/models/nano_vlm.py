"""
NanoVLM: Compact Vision-Language Model using ViTSigLIP and BERT Components

This module provides the updated NanoVLM implementation that leverages the
existing ViTSigLIP and BERT models from the dl-techniques framework through
dedicated VisionEncoder and TextDecoder components.

The architecture combines:
- VisionEncoder: ViTSigLIP-based visual feature extraction
- TextDecoder: BERT-style embeddings with causal transformer layers
- ModalityProjection: Cross-modal alignment between vision and language
- Output projection: Vocabulary prediction head for text generation
"""

import keras
from keras import ops
from typing import Dict, Optional, Tuple, Union, Any, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..utils.logger import logger
from .text_decoder import TextDecoder
from .vision_encoder import VisionEncoder
from ..layers.modality_projection import ModalityProjection

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class NanoVLM(keras.Model):
    """
    NanoVLM: Compact Vision-Language Model using ViTSigLIP and BERT components.

    A lightweight vision-language model that combines ViTSigLIP-based vision
    encoding with BERT-style text decoding through efficient modality projection.
    The model processes images and text tokens to generate coherent text responses.

    The architecture consists of:
    - Vision encoder: VisionEncoder wrapping ViTSigLIP for visual feature extraction
    - Modality projection: Cross-modal alignment layer between vision and language
    - Text decoder: TextDecoder using BERT embeddings with causal attention
    - Output projection: Vocabulary prediction head for text generation

    Args:
        vision_config: Configuration dictionary for vision encoder. Should contain
            keys like 'img_size', 'patch_size', 'embed_dim', 'depth', 'num_heads', etc.
        language_config: Configuration dictionary for text decoder. Should contain
            keys like 'vocab_size', 'hidden_dim', 'num_layers', 'num_heads', 'mlp_dim', etc.
        projection_config: Configuration dictionary for modality projection. Should contain
            keys like 'input_dim', 'output_dim', 'scale_factor', 'use_gelu', etc.
        vocab_size: Size of the vocabulary for text embeddings and output projection.
            Default is 32000 following SmolLM2 conventions.
        dropout_rate: Global dropout rate applied throughout the model. Can be
            overridden by component-specific dropout settings.
        **kwargs: Additional keyword arguments passed to the parent Model class.

    Raises:
        ValueError: If configuration dictionaries are missing required keys or
            contain invalid values.

    Example:
        >>> vision_config = {
        ...     'img_size': 224, 'patch_size': 16, 'embed_dim': 768,
        ...     'depth': 12, 'num_heads': 12, 'mlp_ratio': 4.0
        ... }
        >>> language_config = {
        ...     'vocab_size': 32000, 'hidden_dim': 768, 'num_layers': 12,
        ...     'num_heads': 12, 'mlp_dim': 3072, 'dropout': 0.1
        ... }
        >>> projection_config = {
        ...     'input_dim': 768, 'output_dim': 768, 'scale_factor': 2
        ... }
        >>> model = NanoVLM(vision_config, language_config, projection_config)
    """

    def __init__(
            self,
            vision_config: Dict[str, Any],
            language_config: Dict[str, Any],
            projection_config: Dict[str, Any],
            vocab_size: int = 32000,
            dropout_rate: float = 0.1,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)

        # Store configurations and parameters first
        self.vision_config = vision_config.copy()
        self.language_config = language_config.copy()
        self.projection_config = projection_config.copy()
        self.vocab_size = vocab_size
        self.dropout_rate = dropout_rate

        # Validate configurations
        # FIX 1: Pass the instance's config copies to be validated and potentially modified.
        self._validate_configs(self.vision_config, self.language_config, self.projection_config)

        # Store build input shape for serialization
        self._build_input_shape = None

        # Initialize component placeholders (will be created in build())
        self.vision_encoder = None
        self.modality_projection = None
        self.text_decoder = None
        self.output_projection = None

    def _validate_configs(
            self,
            vision_config: Dict[str, Any],
            language_config: Dict[str, Any],
            projection_config: Dict[str, Any]
    ) -> None:
        """
        Validate configuration dictionaries.

        Args:
            vision_config: Vision encoder configuration
            language_config: Text decoder configuration
            projection_config: Modality projection configuration

        Raises:
            ValueError: If required keys are missing or values are invalid
        """
        # Required vision config keys
        vision_required = ['img_size', 'patch_size', 'embed_dim', 'depth', 'num_heads']
        for key in vision_required:
            if key not in vision_config:
                raise ValueError(f"Missing required vision config key: {key}")

        # Required language config keys
        language_required = ['vocab_size', 'hidden_dim', 'num_layers', 'num_heads', 'mlp_dim']
        for key in language_required:
            if key not in language_config:
                raise ValueError(f"Missing required language config key: {key}")

        # Required projection config keys
        projection_required = ['input_dim', 'output_dim']
        for key in projection_required:
            if key not in projection_config:
                raise ValueError(f"Missing required projection config key: {key}")

        # Validate dimension alignment
        if vision_config['embed_dim'] != projection_config['input_dim']:
            raise ValueError(
                f"Vision embed_dim ({vision_config['embed_dim']}) must match "
                f"projection input_dim ({projection_config['input_dim']})"
            )

        if language_config['hidden_dim'] != projection_config['output_dim']:
            raise ValueError(
                f"Language hidden_dim ({language_config['hidden_dim']}) must match "
                f"projection output_dim ({projection_config['output_dim']})"
            )

        # Ensure vocab_size consistency
        if language_config['vocab_size'] != self.vocab_size:
            logger.warning(
                f"Language config vocab_size ({language_config['vocab_size']}) "
                f"differs from model vocab_size ({self.vocab_size}). "
                f"Using model vocab_size."
            )
            language_config['vocab_size'] = self.vocab_size

    def build(self, input_shape: Optional[Tuple] = None) -> None:
        """
        Build all model components.

        Args:
            input_shape: Input shape tuple (optional, not used for this model)
        """
        if self.built:
            return

        # Store for serialization
        self._build_input_shape = input_shape

        logger.info("Building NanoVLM components...")

        # Build vision encoder
        self.vision_encoder = VisionEncoder(**self.vision_config, name='vision_encoder')

        # Build modality projection
        self.modality_projection = ModalityProjection(**self.projection_config, name='modality_projection')

        # Build text decoder with consistent vocab_size
        text_decoder_config = self.language_config.copy()
        text_decoder_config['vocab_size'] = self.vocab_size
        self.text_decoder = TextDecoder(**text_decoder_config, name='text_decoder')

        # Build output projection
        self.output_projection = keras.layers.Dense(
            self.vocab_size,
            use_bias=False,
            kernel_initializer='glorot_uniform',
            name='output_projection'
        )

        super().build(input_shape)
        logger.info("NanoVLM components built successfully")

    def call(
            self,
            inputs: Union[Dict[str, keras.KerasTensor], Tuple[keras.KerasTensor, keras.KerasTensor]],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through NanoVLM.

        Args:
            inputs: Either a dictionary containing 'images' and 'text_tokens' keys,
                or a tuple of (images, text_tokens). Images should have shape
                [batch, height, width, channels] and text_tokens should have shape
                [batch, sequence_length].
            training: Boolean indicating whether the layer should behave in training
                mode or inference mode. If None, defaults to the current Keras
                learning phase.

        Returns:
            Language model logits of shape [batch, combined_seq_len, vocab_size]
            where combined_seq_len includes both vision tokens and text tokens.

        Raises:
            ValueError: If inputs format is invalid or shapes are incompatible.
        """
        if not self.built:
            self.build()

        # Parse inputs
        if isinstance(inputs, dict):
            if 'images' not in inputs or 'text_tokens' not in inputs:
                raise ValueError("Input dict must contain 'images' and 'text_tokens' keys")
            images = inputs['images']
            text_tokens = inputs['text_tokens']
            token_type_ids = inputs.get('token_type_ids', None)
            position_ids = inputs.get('position_ids', None)
            attention_mask = inputs.get('attention_mask', None)
        elif isinstance(inputs, (tuple, list)) and len(inputs) == 2:
            images, text_tokens = inputs
            token_type_ids = None
            position_ids = None
            attention_mask = None
        else:
            raise ValueError(
                "Inputs must be either a dict with 'images' and 'text_tokens' keys "
                "or a tuple/list of (images, text_tokens)"
            )

        # Process images through vision encoder
        vision_features = self.vision_encoder(images, training=training)

        # FIX 2: WORKAROUND for component mismatch. The ViTSigLIP-based vision encoder
        # returns only patch features (e.g., 196), but the ModalityProjection appears
        # to expect an additional [CLS] token (total 197) to correctly identify the
        # spatial tokens. We add a dummy token to satisfy this shape requirement.
        cls_token_shape = (
            ops.shape(vision_features)[0], 1, ops.shape(vision_features)[2]
        )
        dummy_cls_token = ops.zeros(cls_token_shape, dtype=vision_features.dtype)
        vision_features_with_cls = ops.concatenate(
            [dummy_cls_token, vision_features], axis=1
        )

        # Project visual features to language space
        vision_embeddings = self.modality_projection(vision_features_with_cls, training=training)

        # Get text embeddings and hidden states from decoder
        text_hidden_states = self.text_decoder(
            inputs=text_tokens,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            training=training
        )

        # Combine modalities (concatenate along sequence dimension)
        combined_hidden_states = ops.concatenate([vision_embeddings, text_hidden_states], axis=1)

        # Output projection to vocabulary
        logits = self.output_projection(combined_hidden_states)

        return logits

    def generate(
            self,
            image: keras.KerasTensor,
            prompt_tokens: keras.KerasTensor,
            max_length: int = 100,
            temperature: float = 1.0,
            top_k: int = 50,
            eos_token_id: int = 2
    ) -> keras.KerasTensor:
        """
        Generate text autoregressively given image and prompt.

        Args:
            image: Input image tensor of shape [1, height, width, channels]
            prompt_tokens: Initial prompt tokens of shape [1, prompt_length]
            max_length: Maximum number of tokens to generate
            temperature: Sampling temperature for controlling randomness.
                Higher values make output more random, lower values more deterministic.
            top_k: Number of highest probability tokens to consider for sampling.
                Set to 0 to disable top-k sampling.
            eos_token_id: Token ID that signals end of sequence

        Returns:
            Generated token sequence of shape [1, total_length] where total_length
            includes the original prompt plus generated tokens.

        Note:
            This method assumes batch size of 1 for simplicity. For batch generation,
            use the call method directly with appropriate masking.
        """
        if not self.built:
            self.build()

        # Process image once (cached for generation loop)
        vision_features = self.vision_encoder(image, training=False)
        cls_token_shape = (
            ops.shape(vision_features)[0], 1, ops.shape(vision_features)[2]
        )
        dummy_cls_token = ops.zeros(cls_token_shape, dtype=vision_features.dtype)
        vision_features_with_cls = ops.concatenate(
            [dummy_cls_token, vision_features], axis=1
        )
        vision_embeddings = self.modality_projection(vision_features_with_cls, training=False)

        # Initialize with prompt
        current_tokens = prompt_tokens

        for step in range(max_length):
            # Get current text hidden states
            text_hidden_states = self.text_decoder(
                inputs=current_tokens,
                training=False
            )

            # Combine with vision embeddings
            combined_hidden_states = ops.concatenate([vision_embeddings, text_hidden_states], axis=1)

            # Get logits from output projection
            logits = self.output_projection(combined_hidden_states)

            # Sample next token from the last text position
            # Note: vision tokens come first, so we need to offset by vision sequence length
            vision_seq_len = ops.shape(vision_embeddings)[1]
            text_logits = logits[:, vision_seq_len:, :]  # Extract text logits only
            next_token = self._sample_next_token(text_logits[0, -1, :], temperature, top_k)

            # Append to sequence
            current_tokens = ops.concatenate([
                current_tokens,
                ops.reshape(next_token, [1, 1])
            ], axis=1)

            # Check for end of sequence
            if ops.equal(next_token, eos_token_id):
                break

        return current_tokens

    def _sample_next_token(
            self,
            logits: keras.KerasTensor,
            temperature: float,
            top_k: int
    ) -> keras.KerasTensor:
        """
        Sample next token from logits using temperature and top-k sampling.

        Args:
            logits: Logits tensor of shape [vocab_size]
            temperature: Sampling temperature
            top_k: Top-k parameter for sampling

        Returns:
            Sampled token ID as scalar tensor
        """
        # Apply temperature scaling
        if temperature != 1.0:
            logits = logits / temperature

        # Top-k sampling
        if top_k > 0:
            # Get top-k values and indices
            top_k_logits, top_k_indices = ops.top_k(logits, k=top_k)

            # Sample from top-k distribution
            probs = ops.softmax(top_k_logits)
            # FIX 3: Use keras.random.categorical instead of ops.random.categorical
            sampled_index = keras.random.categorical(
                ops.expand_dims(probs, 0),
                num_samples=1
            )[0, 0]

            # Map back to original vocabulary
            next_token = top_k_indices[sampled_index]
        else:
            # Greedy sampling (argmax)
            next_token = ops.argmax(logits, axis=-1)

        return next_token

    def compute_output_shape(self, input_shape: Tuple) -> Tuple:
        """
        Compute output shape given input shape.

        Args:
            input_shape: Input shape tuple

        Returns:
            Output shape tuple [batch, combined_seq_len, vocab_size]
        """
        if isinstance(input_shape, dict):
            batch_size = input_shape['images'][0]
            text_seq_len = input_shape['text_tokens'][1]
        else:
            # Assume tuple format (images_shape, text_tokens_shape)
            batch_size = input_shape[0][0]
            text_seq_len = input_shape[1][1]

        if batch_size is None or text_seq_len is None:
            return (batch_size, None, self.vocab_size)

        # FIX: Correctly calculate vision sequence length after projection
        vision_patches = (self.vision_config['img_size'] // self.vision_config['patch_size']) ** 2
        scale_factor = self.projection_config.get('scale_factor', 1)
        shuffled_vision_patches = vision_patches // (scale_factor ** 2)
        projected_vision_seq_len = shuffled_vision_patches + 1 # Add 1 for the dummy CLS token

        combined_seq_len = projected_vision_seq_len + text_seq_len

        return (batch_size, combined_seq_len, self.vocab_size)

    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration for serialization.

        Returns:
            Dictionary containing all configuration parameters needed to
            reconstruct the model.
        """
        config = super().get_config()
        config.update({
            "vision_config": self.vision_config,
            "language_config": self.language_config,
            "projection_config": self.projection_config,
            "vocab_size": self.vocab_size,
            "dropout_rate": self.dropout_rate,
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """
        Get build configuration for serialization.

        Returns:
            Dictionary containing build-time configuration.
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """
        Build model from saved configuration.

        Args:
            config: Build configuration dictionary from get_build_config()
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])
        else:
            self.build()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'NanoVLM':
        """
        Create model instance from configuration.

        Args:
            config: Configuration dictionary from get_config()

        Returns:
            New NanoVLM model instance
        """
        return cls(**config)


# ---------------------------------------------------------------------
# Model configurations for different nanoVLM variants
# ---------------------------------------------------------------------

NANOVLM_CONFIGS = {
    "nanovlm_mini": {
        "vision_config": {
            'img_size': 224,
            'patch_size': 16,
            'embed_dim': 384,
            'depth': 6,
            'num_heads': 6,
            'mlp_ratio': 4.0,
            'dropout': 0.1,
        },
        "language_config": {
            'vocab_size': 32000,
            'hidden_dim': 384,
            'num_layers': 6,
            'num_heads': 6,
            'mlp_dim': 1536,
            'dropout': 0.1,
        },
        "projection_config": {
            'input_dim': 384,
            'output_dim': 384,
            'scale_factor': 2,
        },
        "total_params": "16M"
    },

    "nanovlm_base": {
        "vision_config": {
            'img_size': 224,
            'patch_size': 16,
            'embed_dim': 512,
            'depth': 8,
            'num_heads': 8,
            'mlp_ratio': 4.0,
            'dropout': 0.1,
        },
        "language_config": {
            'vocab_size': 32000,
            'hidden_dim': 512,
            'num_layers': 8,
            'num_heads': 8,
            'mlp_dim': 2048,
            'dropout': 0.1,
        },
        "projection_config": {
            'input_dim': 512,
            'output_dim': 512,
            'scale_factor': 2,
        },
        "total_params": "64M"
    },

    "nanovlm_222m": {
        "vision_config": {
            'img_size': 224,
            'patch_size': 16,
            'embed_dim': 768,
            'depth': 12,
            'num_heads': 12,
            'mlp_ratio': 4.0,
            'dropout': 0.0,
        },
        "language_config": {
            'vocab_size': 32000,
            'hidden_dim': 768,
            'num_layers': 12,
            'num_heads': 12,
            'mlp_dim': 3072,
            'dropout': 0.1,
        },
        "projection_config": {
            'input_dim': 768,
            'output_dim': 768,
            'scale_factor': 2,
        },
        "total_params": "222M"
    }
}


# ---------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------

def _prepare_configs(
        vision_config: Dict[str, Any],
        language_config: Dict[str, Any],
        projection_config: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Prepare and enhance model configurations with default values.

    Args:
        vision_config: Vision encoder configuration
        language_config: Text decoder configuration
        projection_config: Modality projection configuration

    Returns:
        Tuple of enhanced configuration dictionaries
    """
    # Enhance projection config with defaults
    enhanced_projection_config = projection_config.copy()
    enhanced_projection_config.setdefault('use_gelu', True)
    enhanced_projection_config.setdefault('use_layer_norm', True)

    return vision_config.copy(), language_config.copy(), enhanced_projection_config


def create_nanovlm(
        variant: str = "nanovlm_base",
        vocab_size: int = 32000,
        dropout_rate: Optional[float] = None
) -> NanoVLM:
    """
    Create NanoVLM model with specified variant configuration.

    Args:
        variant: Model variant name. Must be one of: 'nanovlm_mini', 'nanovlm_base',
            or 'nanovlm_222m'. Defaults to 'nanovlm_base'.
        vocab_size: Size of the vocabulary for text embeddings and output projection.
            Defaults to 32000.
        dropout_rate: Global dropout rate. If None, uses variant-specific defaults.

    Returns:
        Configured NanoVLM model ready for training or inference

    Raises:
        ValueError: If variant is not recognized

    Example:
        >>> # Create different model variants
        >>> mini_model = create_nanovlm("nanovlm_mini")
        >>> base_model = create_nanovlm("nanovlm_base")
        >>> large_model = create_nanovlm("nanovlm_222m")
        >>>
        >>> # Customize parameters
        >>> custom_model = create_nanovlm("nanovlm_base", vocab_size=50000, dropout_rate=0.2)
    """
    if variant not in NANOVLM_CONFIGS:
        available_variants = list(NANOVLM_CONFIGS.keys())
        raise ValueError(
            f"Unknown variant '{variant}'. Available variants: {available_variants}"
        )

    config = NANOVLM_CONFIGS[variant]

    # Prepare enhanced configurations
    vision_config, language_config, projection_config = _prepare_configs(
        config["vision_config"],
        config["language_config"],
        config["projection_config"]
    )

    # Use variant-specific dropout if not explicitly provided
    if dropout_rate is None:
        dropout_rate = language_config.get('dropout', 0.1)

    logger.info(
        f"Creating {variant} with ~{config['total_params']} parameters "
        f"(vocab_size={vocab_size}, dropout_rate={dropout_rate})"
    )

    return NanoVLM(
        vision_config=vision_config,
        language_config=language_config,
        projection_config=projection_config,
        vocab_size=vocab_size,
        dropout_rate=dropout_rate
    )


def create_nanovlm_mini() -> NanoVLM:
    """
    Create NanoVLM-Mini model with compact configuration.

    Creates the smallest vision-language model variant with approximately 16M parameters,
    optimized for resource-constrained environments while maintaining basic
    vision-language understanding capabilities.

    Returns:
        Configured NanoVLM model ready for training or inference

    Example:
        >>> model = create_nanovlm_mini()
        >>> model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        >>> # Ideal for mobile deployment or quick prototyping
    """
    return create_nanovlm("nanovlm_mini")


def create_nanovlm_base() -> NanoVLM:
    """
    Create NanoVLM-Base model with balanced configuration.

    Creates a medium-sized vision-language model with approximately 64M parameters,
    providing a good balance between model capability and computational efficiency.
    This is the recommended variant for most applications.

    Returns:
        Configured NanoVLM model ready for training or inference

    Example:
        >>> model = create_nanovlm_base()
        >>> model.compile(optimizer='adamw', loss='sparse_categorical_crossentropy')
        >>> # Good balance of performance and efficiency
    """
    return create_nanovlm("nanovlm_base")


def create_nanovlm_222m() -> NanoVLM:
    """
    Create NanoVLM-222M model with full configuration.

    Creates the largest vision-language model variant with approximately 222M parameters,
    suitable for applications requiring high-quality vision-language understanding
    where computational resources are less constrained.

    Returns:
        Configured NanoVLM model ready for training or inference

    Example:
        >>> model = create_nanovlm_222m()
        >>> model.compile(optimizer='adamw', loss='sparse_categorical_crossentropy')
        >>> # Best performance for vision-language tasks
    """
    return create_nanovlm("nanovlm_222m")


def get_available_variants() -> List[str]:
    """
    Get list of available NanoVLM model variants.

    Returns:
        List of available variant names that can be used with create_nanovlm()

    Example:
        >>> variants = get_available_variants()
        >>> print(f"Available variants: {variants}")
        >>> # Output: Available variants: ['nanovlm_mini', 'nanovlm_base', 'nanovlm_222m']
    """
    return list(NANOVLM_CONFIGS.keys())


def get_variant_info(variant: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific model variant.

    Args:
        variant: Model variant name

    Returns:
        Dictionary containing variant configuration and metadata

    Raises:
        ValueError: If variant is not recognized

    Example:
        >>> info = get_variant_info("nanovlm_base")
        >>> print(f"Parameters: {info['total_params']}")
        >>> print(f"Hidden dim: {info['language_config']['hidden_dim']}")
    """
    if variant not in NANOVLM_CONFIGS:
        available_variants = list(NANOVLM_CONFIGS.keys())
        raise ValueError(
            f"Unknown variant '{variant}'. Available variants: {available_variants}"
        )

    return NANOVLM_CONFIGS[variant].copy()