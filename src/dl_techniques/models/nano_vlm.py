"""
NanoVLM: A Compact Vision-Language Model

This module implements NanoVLM, a multi-modal model that integrates a vision
encoder and a text decoder to perform vision-language tasks. The model is
designed to be serializable, configurable, and efficient.

Key architectural features:
1.  **Component-based Design**: Built using modular `VisionEncoder`,
    `ModalityProjection`, and `TextDecoder` components.
2.  **Modern Keras 3 Pattern**: All sub-layers are created in `__init__`
    to ensure proper weight serialization and model loading.
3.  **Visual-Text Fusion**: Visual features are processed and projected to serve
    as a prefix to the text decoder's input sequence.
4.  **Combined Attention Mask**: A custom attention mask is generated on the fly
    to allow full attention over visual tokens and causal attention over text tokens.
5.  **Autoregressive Generation**: Includes a `generate` method for text generation
    conditioned on an image and a text prompt, with support for temperature
    and top-k sampling.
"""

import keras
from keras import ops
from typing import Optional, Dict, Any, Tuple, Union, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..utils.logger import logger
from .vision_encoder import VisionEncoder
from .text_decoder import TextDecoder
from ..layers.modality_projection import ModalityProjection


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class NanoVLM(keras.Model):
    """
    NanoVLM model integrating vision and language capabilities.

    This model implements a compact vision-language model that combines a vision
    encoder with a text decoder to perform multimodal tasks like image captioning
    and visual question answering.

    The architecture follows the modern Keras 3 pattern where all sub-components
    are created in __init__ for proper serialization support.

    Args:
        vision_config: Dictionary containing configuration for VisionEncoder.
            Required keys: 'embed_dim'. Common keys include 'img_size', 'patch_size',
            'depth', 'num_heads', 'mlp_ratio', 'dropout'.
        language_config: Dictionary containing configuration for TextDecoder.
            Required keys: 'hidden_dim', 'vocab_size'. Common keys include 'num_layers',
            'num_heads', 'mlp_dim', 'dropout'.
        projection_config: Dictionary containing configuration for ModalityProjection.
            Required keys: 'input_dim', 'output_dim'. Common keys include 'scale_factor',
            'use_gelu', 'use_layer_norm'.
        vocab_size: Optional vocabulary size override. If provided, overrides the
            vocab_size in language_config. Must be positive.
        dropout_rate: Optional global dropout rate override. Must be between 0 and 1.
            If provided, applies to both vision and language components.
        **kwargs: Additional keyword arguments for the keras.Model base class.

    Input shape:
        Dictionary or tuple containing:
        - 'images': Tensor of shape (batch_size, height, width, channels)
        - 'text_tokens': Tensor of shape (batch_size, sequence_length)
        - Optional: 'token_type_ids', 'position_ids', 'attention_mask'

    Output shape:
        Tensor of shape (batch_size, combined_sequence_length, vocab_size)
        where combined_sequence_length = projected_vision_length + text_length

    Raises:
        ValueError: If any configuration is invalid or incompatible.
        TypeError: If configuration parameters have wrong types.

    Example:
        ```python
        # Basic usage with configuration dictionaries
        vision_config = {
            'img_size': 224, 'patch_size': 16, 'embed_dim': 512,
            'depth': 8, 'num_heads': 8, 'mlp_ratio': 4.0
        }
        language_config = {
            'vocab_size': 32000, 'hidden_dim': 512, 'num_layers': 8,
            'num_heads': 8, 'mlp_dim': 2048
        }
        projection_config = {
            'input_dim': 512, 'output_dim': 512, 'scale_factor': 2
        }

        model = NanoVLM(
            vision_config=vision_config,
            language_config=language_config,
            projection_config=projection_config
        )

        # Using with image and text inputs
        images = keras.random.normal((2, 224, 224, 3))
        text_tokens = keras.random.uniform((2, 50), 0, 32000, dtype='int32')

        outputs = model({'images': images, 'text_tokens': text_tokens})
        print(outputs.shape)  # (2, combined_seq_len, 32000)

        # Autoregressive generation
        generated = model.generate(
            image=images[:1],  # Single image
            prompt_tokens=text_tokens[:1, :10],  # Short prompt
            max_length=50,
            temperature=0.8
        )
        ```

    Note:
        This implementation follows the modern Keras 3 pattern where all sub-layers
        are created in __init__ and Keras handles the building automatically. This
        ensures proper serialization and eliminates common build errors.
    """

    def __init__(
        self,
        vision_config: Dict[str, Any],
        language_config: Dict[str, Any],
        projection_config: Dict[str, Any],
        vocab_size: Optional[int] = None,
        dropout_rate: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate input types
        if not isinstance(vision_config, dict):
            raise TypeError(f"vision_config must be a dictionary, got {type(vision_config)}")
        if not isinstance(language_config, dict):
            raise TypeError(f"language_config must be a dictionary, got {type(language_config)}")
        if not isinstance(projection_config, dict):
            raise TypeError(f"projection_config must be a dictionary, got {type(projection_config)}")

        if vocab_size is not None and (not isinstance(vocab_size, int) or vocab_size <= 0):
            raise ValueError(f"vocab_size must be a positive integer, got {vocab_size}")

        if dropout_rate is not None and (not isinstance(dropout_rate, (int, float)) or not 0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

        # Validate and prepare configurations
        self._validate_configs(vision_config, language_config, projection_config)

        # Handle vocab_size override
        if vocab_size is not None:
            if 'vocab_size' in language_config and language_config['vocab_size'] != vocab_size:
                logger.warning(
                    f"Overriding language_config vocab_size {language_config['vocab_size']} "
                    f"with model-level vocab_size {vocab_size}."
                )
            language_config = language_config.copy()  # Don't modify original
            language_config['vocab_size'] = vocab_size

        # Handle dropout_rate override
        if dropout_rate is not None:
            vision_config = vision_config.copy()  # Don't modify original
            language_config = language_config.copy()
            vision_config['dropout'] = dropout_rate
            language_config['dropout'] = dropout_rate

        # Store configuration for serialization
        self.vision_config = vision_config
        self.language_config = language_config
        self.projection_config = projection_config
        self.vocab_size = self.language_config['vocab_size']
        self.dropout_rate = dropout_rate if dropout_rate is not None else self.language_config.get('dropout', 0.1)

        # CREATE all sub-layers in __init__ following modern Keras 3 pattern
        try:
            self.vision_encoder = VisionEncoder(**self.vision_config, name="vision_encoder")
            self.modality_projection = ModalityProjection(**self.projection_config, name="modality_projection")
            self.text_decoder = TextDecoder(**self.language_config, name="text_decoder")
            self.output_projection = keras.layers.Dense(
                self.vocab_size,
                use_bias=False,
                name="output_projection"
            )
        except Exception as e:
            logger.error(f"Failed to create NanoVLM sub-components: {e}")
            raise ValueError(f"Failed to create NanoVLM. This might be due to incompatible configurations. Original error: {e}")

        logger.info(f"Initialized NanoVLM with vocab_size={self.vocab_size}, dropout={self.dropout_rate}")

    def _validate_configs(
        self,
        vision_config: Dict[str, Any],
        language_config: Dict[str, Any],
        projection_config: Dict[str, Any]
    ) -> None:
        """
        Validate that the configurations are complete and compatible.

        Args:
            vision_config: Vision encoder configuration dictionary.
            language_config: Language decoder configuration dictionary.
            projection_config: Projection layer configuration dictionary.

        Raises:
            ValueError: If any required keys are missing or configurations are incompatible.
        """
        # Check for required keys
        vision_required = ['embed_dim']
        language_required = ['hidden_dim', 'vocab_size']
        projection_required = ['input_dim', 'output_dim']

        for key in vision_required:
            if key not in vision_config:
                raise ValueError(f"Missing required vision config key: '{key}'")

        for key in language_required:
            if key not in language_config:
                raise ValueError(f"Missing required language config key: '{key}'")

        for key in projection_required:
            if key not in projection_config:
                raise ValueError(f"Missing required projection config key: '{key}'")

        # Validate value types and ranges
        if not isinstance(vision_config['embed_dim'], int) or vision_config['embed_dim'] <= 0:
            raise ValueError(f"vision_config['embed_dim'] must be a positive integer, got {vision_config['embed_dim']}")

        if not isinstance(language_config['hidden_dim'], int) or language_config['hidden_dim'] <= 0:
            raise ValueError(f"language_config['hidden_dim'] must be a positive integer, got {language_config['hidden_dim']}")

        if not isinstance(language_config['vocab_size'], int) or language_config['vocab_size'] <= 0:
            raise ValueError(f"language_config['vocab_size'] must be a positive integer, got {language_config['vocab_size']}")

        # Check for dimension consistency
        if vision_config['embed_dim'] != projection_config['input_dim']:
            raise ValueError(
                f"Vision embed_dim ({vision_config['embed_dim']}) must match "
                f"projection input_dim ({projection_config['input_dim']})."
            )

        if language_config['hidden_dim'] != projection_config['output_dim']:
            raise ValueError(
                f"Language hidden_dim ({language_config['hidden_dim']}) must match "
                f"projection output_dim ({projection_config['output_dim']})."
            )

    def call(
        self,
        inputs: Union[Dict[str, keras.KerasTensor], Tuple[keras.KerasTensor, ...]],
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass for the NanoVLM model.

        Args:
            inputs: Input data in one of two formats:
                - Dictionary with 'images' and 'text_tokens' keys, plus optional
                  'token_type_ids', 'position_ids', 'attention_mask'
                - Tuple/list of (images, text_tokens)
            training: Boolean indicating training or inference mode.

        Returns:
            Logits tensor of shape (batch_size, combined_seq_length, vocab_size).

        Raises:
            ValueError: If inputs are in wrong format or missing required keys.
            TypeError: If inputs have wrong types.
        """
        # Parse inputs
        if isinstance(inputs, dict):
            if not ('images' in inputs and 'text_tokens' in inputs):
                missing_keys = [key for key in ['images', 'text_tokens'] if key not in inputs]
                raise ValueError(f"Input dict must contain 'images' and 'text_tokens' keys. Missing: {missing_keys}")

            images, text_tokens = inputs['images'], inputs['text_tokens']
            text_kwargs = {k: v for k, v in inputs.items() if k not in ['images', 'text_tokens']}

        elif isinstance(inputs, (tuple, list)):
            if len(inputs) != 2:
                raise ValueError(
                    f"Inputs tuple/list must contain exactly 2 items (images, text_tokens). "
                    f"Received {len(inputs)} items."
                )
            images, text_tokens = inputs
            text_kwargs = {}
        else:
            raise TypeError(
                f"Inputs must be either a dict with 'images' and 'text_tokens' keys "
                f"or a tuple/list of (images, text_tokens). Got {type(inputs)}"
            )

        # Vision feature extraction
        vision_features = self.vision_encoder(images, training=training)
        batch_size = ops.shape(vision_features)[0]
        vision_dim = ops.shape(vision_features)[-1]

        # Prepend a dummy CLS token for ModalityProjection PixelShuffle compatibility
        dummy_cls = ops.zeros((batch_size, 1, vision_dim), dtype=vision_features.dtype)
        vision_features_with_cls = ops.concatenate([dummy_cls, vision_features], axis=1)
        projected_vision_features = self.modality_projection(vision_features_with_cls, training=training)

        # Text feature extraction
        text_embeddings = self.text_decoder.embeddings(
            input_ids=text_tokens,
            token_type_ids=text_kwargs.get("token_type_ids"),
            position_ids=text_kwargs.get("position_ids"),
            training=training
        )

        # Multimodal fusion
        vision_seq_len = ops.shape(projected_vision_features)[1]
        text_seq_len = ops.shape(text_embeddings)[1]

        combined_embeddings = ops.concatenate([projected_vision_features, text_embeddings], axis=1)

        # Create combined attention mask: full attention for vision, causal for text
        attention_mask = self._create_attention_mask(
            vision_seq_len,
            text_seq_len,
            dtype=combined_embeddings.dtype
        )

        # Pass through decoder layers
        hidden_states = combined_embeddings
        for decoder_layer in self.text_decoder.decoder_layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                training=training
            )

        hidden_states = self.text_decoder.final_norm(hidden_states, training=training)
        logits = self.output_projection(hidden_states)

        return logits

    def _create_attention_mask(
        self,
        vision_seq_len: int,
        text_seq_len: int,
        dtype: str
    ) -> keras.KerasTensor:
        """
        Creates a combined attention mask for vision (full) and text (causal).

        Args:
            vision_seq_len: Length of vision sequence.
            text_seq_len: Length of text sequence.
            dtype: Data type for the mask tensor.

        Returns:
            Attention mask tensor of shape (1, 1, total_seq_len, total_seq_len).
        """
        total_seq_len = vision_seq_len + text_seq_len

        # Create causal mask for the entire sequence
        causal_mask = ops.tril(ops.ones((total_seq_len, total_seq_len), dtype=dtype))

        # Vision part can attend to all vision tokens (full attention)
        causal_mask = ops.slice_update(
            causal_mask,
            (0, 0),
            ops.ones((vision_seq_len, vision_seq_len), dtype=dtype)
        )

        # Text can attend to all vision tokens
        causal_mask = ops.slice_update(
            causal_mask,
            (vision_seq_len, 0),
            ops.ones((text_seq_len, vision_seq_len), dtype=dtype)
        )

        # Reshape for attention layer: (batch_size, 1, seq_len, seq_len)
        return ops.expand_dims(ops.expand_dims(causal_mask, 0), 0)

    def generate(
        self,
        image: keras.KerasTensor,
        prompt_tokens: keras.KerasTensor,
        max_length: int = 50,
        eos_token_id: int = 2,
        temperature: float = 1.0,
        top_k: int = 50
    ) -> keras.KerasTensor:
        """
        Autoregressively generate text conditioned on an image and a prompt.

        Args:
            image: Input image tensor of shape (1, H, W, C).
            prompt_tokens: Input prompt token IDs of shape (1, prompt_length).
            max_length: Maximum number of tokens to generate. Must be positive.
            eos_token_id: Token ID for end-of-sequence. Must be non-negative.
            temperature: Sampling temperature. Must be positive.
            top_k: Top-k filtering for sampling. Use 0 for greedy decoding.

        Returns:
            Generated token sequence of shape (1, generated_length).

        Raises:
            ValueError: If any parameter is invalid or input shapes are wrong.
        """
        # Validate inputs
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")
        if eos_token_id < 0:
            raise ValueError(f"eos_token_id must be non-negative, got {eos_token_id}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if top_k < 0:
            raise ValueError(f"top_k must be non-negative, got {top_k}")

        # Validate input shapes
        if len(ops.shape(image)) != 4 or ops.shape(image)[0] != 1:
            raise ValueError(f"image must have shape (1, H, W, C), got {ops.shape(image)}")
        if len(ops.shape(prompt_tokens)) != 2 or ops.shape(prompt_tokens)[0] != 1:
            raise ValueError(f"prompt_tokens must have shape (1, seq_len), got {ops.shape(prompt_tokens)}")

        generated_tokens = prompt_tokens

        for _ in range(max_length):
            inputs = {'images': image, 'text_tokens': generated_tokens}
            logits = self(inputs, training=False)

            # Get logits for the last token only
            last_token_logits = logits[:, -1, :]

            # Sample the next token
            next_token = self._sample_next_token(last_token_logits, temperature, top_k)

            # Append the new token
            generated_tokens = ops.concatenate([generated_tokens, next_token[:, None]], axis=1)

            # Check for end-of-sequence
            if next_token[0] == eos_token_id:
                break

        return generated_tokens

    def _sample_next_token(
        self,
        logits: keras.KerasTensor,
        temperature: float,
        top_k: int
    ) -> keras.KerasTensor:
        """
        Samples the next token from logits using temperature and top-k filtering.

        Args:
            logits: Logits tensor of shape (batch_size, vocab_size).
            temperature: Sampling temperature.
            top_k: Top-k filtering parameter.

        Returns:
            Sampled token indices of shape (batch_size,).

        Raises:
            ValueError: If logits have wrong shape.
        """
        if len(ops.shape(logits)) != 2:
            raise ValueError(f"Expected 2D logits, but got shape {ops.shape(logits)}")

        # Greedy decoding for top_k = 0
        if top_k == 0:
            return ops.argmax(logits, axis=-1)

        # Apply temperature scaling
        logits = logits / temperature

        # Top-k sampling
        top_k_logits, top_k_indices = ops.top_k(logits, k=top_k)
        sampled_indices_in_k = keras.random.categorical(top_k_logits, num_samples=1)
        sampled_token = ops.take_along_axis(top_k_indices, sampled_indices_in_k, axis=1)

        return ops.squeeze(sampled_token, axis=1)

    def compute_output_shape(self, input_shape: Union[Dict[str, Tuple], Tuple]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the model.

        Args:
            input_shape: Input shape specification in one of two formats:
                - Dictionary with 'images' and 'text_tokens' keys
                - Tuple of (image_shape, text_shape)

        Returns:
            Output shape tuple (batch_size, combined_seq_length, vocab_size).

        Raises:
            ValueError: If input_shape format is invalid.
        """
        if isinstance(input_shape, dict):
            if 'images' not in input_shape or 'text_tokens' not in input_shape:
                raise ValueError("input_shape dict must contain 'images' and 'text_tokens' keys")
            image_shape, text_shape = input_shape['images'], input_shape['text_tokens']
        elif isinstance(input_shape, (tuple, list)) and len(input_shape) == 2:
            image_shape, text_shape = input_shape
        else:
            raise ValueError(
                "input_shape must be either a dict with 'images'/'text_tokens' keys "
                "or a tuple/list of (image_shape, text_shape)"
            )

        batch_size = image_shape[0]
        text_seq_len = text_shape[1]

        # Calculate projected vision sequence length
        img_size = self.vision_config.get('img_size', 224)
        patch_size = self.vision_config.get('patch_size', 16)
        scale_factor = self.projection_config.get('scale_factor', 1)

        vision_patches = (img_size // patch_size) ** 2
        shuffled_vision_patches = vision_patches // (scale_factor ** 2)
        projected_vision_seq_len = shuffled_vision_patches + 1  # +1 for dummy CLS token

        combined_seq_len = projected_vision_seq_len + text_seq_len
        return (batch_size, combined_seq_len, self.vocab_size)

    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration for serialization.

        Returns:
            Dictionary containing all configuration needed to recreate the model.
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

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "NanoVLM":
        """
        Create model from configuration dictionary.

        Args:
            config: Configuration dictionary from get_config().

        Returns:
            New NanoVLM instance.
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
        vision_config: Vision encoder configuration dictionary.
        language_config: Text decoder configuration dictionary.
        projection_config: Modality projection configuration dictionary.

    Returns:
        Tuple of enhanced configuration dictionaries.
    """
    # Create copies to avoid modifying originals
    enhanced_vision_config = vision_config.copy()
    enhanced_language_config = language_config.copy()
    enhanced_projection_config = projection_config.copy()

    # Add defaults to projection config
    enhanced_projection_config.setdefault('use_gelu', True)
    enhanced_projection_config.setdefault('use_layer_norm', True)

    return enhanced_vision_config, enhanced_language_config, enhanced_projection_config


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
            Must be positive. Defaults to 32000.
        dropout_rate: Global dropout rate. Must be between 0 and 1. If None,
            uses variant-specific defaults.

    Returns:
        Configured NanoVLM model ready for training or inference.

    Raises:
        ValueError: If variant is not recognized or parameters are invalid.

    Example:
        ```python
        # Create different model variants
        mini_model = create_nanovlm("nanovlm_mini")
        base_model = create_nanovlm("nanovlm_base")
        large_model = create_nanovlm("nanovlm_222m")

        # Customize parameters
        custom_model = create_nanovlm("nanovlm_base", vocab_size=50000, dropout_rate=0.2)
        ```
    """
    if variant not in NANOVLM_CONFIGS:
        available_variants = list(NANOVLM_CONFIGS.keys())
        raise ValueError(
            f"Unknown variant '{variant}'. Available variants: {available_variants}"
        )

    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")

    if dropout_rate is not None and not (0.0 <= dropout_rate <= 1.0):
        raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

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
        Configured NanoVLM model ready for training or inference.

    Example:
        ```python
        model = create_nanovlm_mini()
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        # Ideal for mobile deployment or quick prototyping
        ```
    """
    return create_nanovlm("nanovlm_mini")


def create_nanovlm_base() -> NanoVLM:
    """
    Create NanoVLM-Base model with balanced configuration.

    Creates a medium-sized vision-language model with approximately 64M parameters,
    providing a good balance between model capability and computational efficiency.
    This is the recommended variant for most applications.

    Returns:
        Configured NanoVLM model ready for training or inference.

    Example:
        ```python
        model = create_nanovlm_base()
        model.compile(optimizer='adamw', loss='sparse_categorical_crossentropy')
        # Good balance of performance and efficiency
        ```
    """
    return create_nanovlm("nanovlm_base")


def create_nanovlm_222m() -> NanoVLM:
    """
    Create NanoVLM-222M model with full configuration.

    Creates the largest vision-language model variant with approximately 222M parameters,
    suitable for applications requiring high-quality vision-language understanding
    where computational resources are less constrained.

    Returns:
        Configured NanoVLM model ready for training or inference.

    Example:
        ```python
        model = create_nanovlm_222m()
        model.compile(optimizer='adamw', loss='sparse_categorical_crossentropy')
        # Best performance for vision-language tasks
        ```
    """
    return create_nanovlm("nanovlm_222m")


def get_available_variants() -> List[str]:
    """
    Get list of available NanoVLM model variants.

    Returns:
        List of available variant names that can be used with create_nanovlm().

    Example:
        ```python
        variants = get_available_variants()
        print(f"Available variants: {variants}")
        # Output: Available variants: ['nanovlm_mini', 'nanovlm_base', 'nanovlm_222m']
        ```
    """
    return list(NANOVLM_CONFIGS.keys())


def get_variant_info(variant: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific model variant.

    Args:
        variant: Model variant name. Must be one of the available variants.

    Returns:
        Dictionary containing variant configuration and metadata.

    Raises:
        ValueError: If variant is not recognized.

    Example:
        ```python
        info = get_variant_info("nanovlm_base")
        print(f"Parameters: {info['total_params']}")
        print(f"Hidden dim: {info['language_config']['hidden_dim']}")
        ```
    """
    if variant not in NANOVLM_CONFIGS:
        available_variants = list(NANOVLM_CONFIGS.keys())
        raise ValueError(
            f"Unknown variant '{variant}'. Available variants: {available_variants}"
        )

    return NANOVLM_CONFIGS[variant].copy()