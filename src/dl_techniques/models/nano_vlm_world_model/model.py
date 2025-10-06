"""
Score-Based nanoVLM: Navigable World Model Architecture

This module implements the complete score-based vision-language model following
the Miyasawa theorem framework. Instead of deterministic prediction, the model
learns the score field ∇ log p(image, text) and navigates it via diffusion.

Key Innovation:
    The VLM is not a predictor but a world model - an implicit representation
    of the joint probability distribution p(image, text). By learning score
    functions via Denoising Score Matching, we can navigate this semantic
    landscape for generation, understanding, and reasoning.

Protocols Implemented:
    1. Text-to-Image: Conditional diffusion on p(image | text)
    2. Image-to-Text: Latent space diffusion on p(text | image)
    3. Joint Modeling: Unified score field ∇ log p(image, text)
"""

import keras
import numpy as np
from keras import ops, layers
from typing import Dict, Optional, Tuple, Union, Any, Literal

from dl_techniques.utils.logger import logger
from dl_techniques.layers.text_encoder import TextEncoder
from dl_techniques.layers.vision_encoder import create_vision_encoder

from .denoisers import VisionDenoiser, TextDenoiser, JointDenoiser
from .scheduler import DiffusionScheduler

@keras.saving.register_keras_serializable()
class ScoreBasedNanoVLM(keras.Model):
    """
    Score-Based nanoVLM: A Navigable Vision-Language World Model.

    Re-imagines VLMs through Miyasawa's theorem: instead of learning direct
    mappings (text→image or image→text), this model learns the score function
    ∇ log p(image, text) via Denoising Score Matching. All generative tasks
    become navigation of this learned semantic landscape via diffusion.

    **Theoretical Foundation:**
    By Miyasawa's theorem (Tweedie's formula), an optimal denoiser D(x_t, c, t)
    trained via MSE provides the score function:
        ∇_x log p(x_t | c) = (1/σ²) * (D(x_t, c, t) - x_t)

    This transforms VLM training from supervised prediction to learning a
    "physics of meaning" - a vector field defining how concepts attract and
    repel in semantic space.

    **Three Operational Modes:**
    1. **Text-to-Image**: Navigate p(image | text) via reverse diffusion
    2. **Image-to-Text**: Navigate p(text | image) in embedding space
    3. **Joint Reasoning**: Traverse the unified field ∇ log p(image, text)

    Args:
        vision_config: Configuration dict for vision processing.
        text_config: Configuration dict for text processing.
        diffusion_config: Configuration for diffusion scheduler.
        vocab_size: Vocabulary size for text. Defaults to 32000.
        generation_mode: Which generative mode to enable
            ('text_to_image', 'image_to_text', 'joint'). Defaults to 'joint'.
        use_classifier_free_guidance: Enable CFG for stronger conditioning.
            Defaults to True.
        **kwargs: Additional model arguments.

    Example:
        ```python
        # Create score-based VLM
        model = ScoreBasedNanoVLM(
            vision_config={'img_size': 224, 'embed_dim': 768, ...},
            text_config={'vocab_size': 32000, 'embed_dim': 768, ...},
            diffusion_config={'num_timesteps': 1000, 'beta_schedule': 'cosine'},
            generation_mode='joint'
        )

        # Training: Learn to denoise
        with tf.GradientTape() as tape:
            # Add noise to clean data
            noisy_vision, noise, timesteps = scheduler.add_noise(clean_vision, ...)

            # Denoise
            denoised = model.denoise_vision(noisy_vision, text_features, timesteps)

            # DSM loss: ||D(x_t, c) - x_0||²
            loss = mse(denoised, clean_vision)

        # Generation: Navigate score field
        generated_image = model.generate_from_text(text_prompt, num_steps=50)
        generated_text = model.generate_from_image(image, num_steps=50)
        ```
    """

    def __init__(
            self,
            vision_config: Dict[str, Any],
            text_config: Dict[str, Any],
            diffusion_config: Dict[str, Any],
            vocab_size: int = 32000,
            generation_mode: Literal['text_to_image', 'image_to_text', 'joint'] = 'joint',
            use_classifier_free_guidance: bool = True,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)

        # Store configurations
        self.vision_config = vision_config
        self.text_config = text_config
        self.diffusion_config = diffusion_config
        self.vocab_size = vocab_size
        self.generation_mode = generation_mode
        self.use_classifier_free_guidance = use_classifier_free_guidance

        # Validate dimensions
        vision_dim = vision_config.get('embed_dim', 768)
        text_dim = text_config.get('embed_dim', 768)

        logger.info(f"Initializing Score-Based nanoVLM in '{generation_mode}' mode")
        logger.info(f"  Vision dim: {vision_dim}, Text dim: {text_dim}")

        # === Core Encoding Components ===
        # Vision encoder (processes clean images to features)
        self.vision_encoder = create_vision_encoder(**vision_config)

        # Text encoder (processes text to embeddings)
        self.text_encoder = TextEncoder(**text_config, name='text_encoder')

        # === Diffusion Scheduler ===
        self.scheduler = DiffusionScheduler(**diffusion_config, name='scheduler')

        # === Denoiser Networks (The Core Innovation) ===


        if generation_mode in ['text_to_image', 'joint']:
            self.vision_denoiser = VisionDenoiser(
                vision_config=vision_config,
                text_dim=text_dim,
                num_layers=12,
                name='vision_denoiser'
            )

        if generation_mode in ['image_to_text', 'joint']:
            self.text_denoiser = TextDenoiser(
                text_dim=text_dim,
                vision_dim=vision_dim,
                num_layers=12,
                name='text_denoiser'
            )

        if generation_mode == 'joint':
            self.joint_denoiser = JointDenoiser(
                vision_dim=vision_dim,
                text_dim=text_dim,
                hidden_dim=max(vision_dim, text_dim),
                num_layers=16,
                name='joint_denoiser'
            )

        # === Output Heads ===
        # For image-to-text, we need to decode embeddings to tokens
        if generation_mode in ['image_to_text', 'joint']:
            self.text_decoder_head = layers.Dense(
                vocab_size,
                use_bias=False,
                kernel_initializer='glorot_uniform',
                name='text_decoder_head'
            )

        logger.info("Score-Based nanoVLM initialized successfully")

    def build(self, input_shape: Union[Dict, Tuple]) -> None:
        """Build all components."""
        if self.built:
            return

        # Build encoders
        if isinstance(input_shape, dict):
            vision_shape = input_shape.get('images')
            text_shape = input_shape.get('text')
        else:
            vision_shape, text_shape = input_shape

        if vision_shape is not None:
            self.vision_encoder.build(vision_shape)

        if text_shape is not None:
            self.text_encoder.build({'input_ids': text_shape})

        super().build(input_shape)

    def call(
            self,
            inputs: Dict[str, keras.KerasTensor],
            training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """
        Forward pass for training.

        During training, we perform Denoising Score Matching:
        1. Encode clean data to features
        2. Add noise according to diffusion schedule
        3. Denoise and compare to clean target

        Args:
            inputs: Dictionary containing:
                - 'images': Clean images [batch, H, W, C]
                - 'text': Text token IDs [batch, seq_len]
                - 'timesteps': Random timesteps for DSM [batch] (optional)
            training: Training mode flag

        Returns:
            Dictionary with denoised outputs for loss computation
        """
        images = inputs['images']
        text_tokens = inputs.get('text')
        timesteps = inputs.get('timesteps')

        # Encode clean data
        vision_features = self.vision_encoder(images, training=training)

        if text_tokens is not None:
            text_features = self.text_encoder(
                {'input_ids': text_tokens}, training=training
            )
        else:
            text_features = None

        # Sample random timesteps if not provided
        if timesteps is None:
            batch_size = ops.shape(images)[0]
            timesteps = keras.random.uniform(
                (batch_size,), minval=0, maxval=self.scheduler.num_timesteps,
                dtype='int32'
            )

        outputs = {}

        # Add noise and denoise based on mode
        if self.generation_mode in ['text_to_image', 'joint']:
            # Text-to-Image: Denoise vision conditioned on text
            noise_vision = keras.random.normal(ops.shape(vision_features))
            noisy_vision = self.scheduler.add_noise(vision_features, noise_vision, timesteps)

            denoised_vision = self.vision_denoiser(
                noisy_vision, text_features, timesteps, training=training
            )
            outputs['denoised_vision'] = denoised_vision
            outputs['target_vision'] = vision_features
            outputs['noise_vision'] = noise_vision

        if self.generation_mode in ['image_to_text', 'joint']:
            # Image-to-Text: Denoise text embeddings conditioned on vision
            noise_text = keras.random.normal(ops.shape(text_features))
            noisy_text = self.scheduler.add_noise(text_features, noise_text, timesteps)

            denoised_text = self.text_denoiser(
                noisy_text, vision_features, timesteps, training=training
            )
            outputs['denoised_text'] = denoised_text
            outputs['target_text'] = text_features
            outputs['noise_text'] = noise_text

        if self.generation_mode == 'joint':
            # Joint: Denoise both simultaneously
            noise_v = keras.random.normal(ops.shape(vision_features))
            noise_t = keras.random.normal(ops.shape(text_features))
            noisy_v = self.scheduler.add_noise(vision_features, noise_v, timesteps)
            noisy_t = self.scheduler.add_noise(text_features, noise_t, timesteps)

            denoised_v, denoised_t = self.joint_denoiser(
                noisy_v, noisy_t, timesteps, training=training
            )
            outputs['joint_denoised_vision'] = denoised_v
            outputs['joint_denoised_text'] = denoised_t
            outputs['joint_target_vision'] = vision_features
            outputs['joint_target_text'] = text_features

        outputs['timesteps'] = timesteps
        return outputs

    def generate_from_text(
            self,
            text_features: keras.KerasTensor,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            generator: Optional[Any] = None
    ) -> keras.KerasTensor:
        """
        Generate images from text via reverse diffusion (Protocol 1).

        Implements the reverse-time SDE: starting from noise, iteratively
        denoise by following the score field ∇ log p(image | text).

        Args:
            text_features: Text conditioning [batch, seq_len, text_dim]
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance strength (>1 for stronger)
            generator: Random generator for reproducibility

        Returns:
            Generated images [batch, H, W, C]
        """
        if not hasattr(self, 'vision_denoiser'):
            raise ValueError("Model not configured for text-to-image generation")

        batch_size = ops.shape(text_features)[0]

        # Get vision feature shape from encoder
        dummy_img = keras.random.normal((1, 224, 224, 3))
        vision_feat_shape = ops.shape(self.vision_encoder(dummy_img, training=False))
        seq_len, feat_dim = vision_feat_shape[1], vision_feat_shape[2]

        # Start from pure noise
        latent_shape = (batch_size, seq_len, feat_dim)
        latents = keras.random.normal(latent_shape)

        # Timestep schedule for inference
        timesteps = np.linspace(
            self.scheduler.num_timesteps - 1, 0, num_inference_steps, dtype=np.int32
        )

        # Reverse diffusion loop
        for i, t in enumerate(timesteps):
            t_tensor = ops.convert_to_tensor([t] * batch_size, dtype='int32')

            # Predict noise
            if self.use_classifier_free_guidance and guidance_scale != 1.0:
                # Classifier-Free Guidance: interpolate conditional and unconditional
                # Unconditional: use zero/null text features
                null_text = ops.zeros_like(text_features)

                # Concatenate for parallel processing
                latent_input = ops.concatenate([latents, latents], axis=0)
                text_input = ops.concatenate([text_features, null_text], axis=0)
                t_input = ops.concatenate([t_tensor, t_tensor], axis=0)

                noise_pred = self.vision_denoiser(
                    latent_input, text_input, t_input, training=False
                )

                # Split and apply guidance
                noise_cond, noise_uncond = ops.split(noise_pred, 2, axis=0)
                noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            else:
                noise_pred = self.vision_denoiser(
                    latents, text_features, t_tensor, training=False
                )

            # Convert denoised output to noise prediction if needed
            if self.scheduler.prediction_type == 'sample':
                noise_pred = self.scheduler.predict_noise_from_start(
                    latents, t_tensor, noise_pred
                )

            # Take reverse diffusion step
            latents, _ = self.scheduler.step(noise_pred, t, latents)

        # Decode latents to images (this would need a decoder)
        # For now, return the latent representation
        logger.info(f"Generated vision features via {num_inference_steps} diffusion steps")
        return latents

    def generate_from_image(
            self,
            vision_features: keras.KerasTensor,
            num_inference_steps: int = 50,
            max_length: int = 77,
            guidance_scale: float = 3.0
    ) -> keras.KerasTensor:
        """
        Generate text from images via latent diffusion (Protocol 2).

        Instead of autoregressive token-by-token generation, this performs
        holistic generation by denoising a text embedding, then decoding it.
        This avoids error propagation and enables semantic manipulation.

        Args:
            vision_features: Vision conditioning [batch, seq_len, vision_dim]
            num_inference_steps: Number of denoising steps
            max_length: Maximum text sequence length
            guidance_scale: Guidance strength

        Returns:
            Generated text embeddings [batch, max_length, text_dim]
        """
        if not hasattr(self, 'text_denoiser'):
            raise ValueError("Model not configured for image-to-text generation")

        batch_size = ops.shape(vision_features)[0]
        text_dim = self.text_config['embed_dim']

        # Start from noise in text embedding space
        latents = keras.random.normal((batch_size, max_length, text_dim))

        # Inference timestep schedule
        timesteps = np.linspace(
            self.scheduler.num_timesteps - 1, 0, num_inference_steps, dtype=np.int32
        )

        # Reverse diffusion in text space
        for i, t in enumerate(timesteps):
            t_tensor = ops.convert_to_tensor([t] * batch_size, dtype='int32')

            if self.use_classifier_free_guidance and guidance_scale != 1.0:
                null_vision = ops.zeros_like(vision_features)
                latent_input = ops.concatenate([latents, latents], axis=0)
                vision_input = ops.concatenate([vision_features, null_vision], axis=0)
                t_input = ops.concatenate([t_tensor, t_tensor], axis=0)

                text_pred = self.text_denoiser(
                    latent_input, vision_input, t_input, training=False
                )

                text_cond, text_uncond = ops.split(text_pred, 2, axis=0)
                text_pred = text_uncond + guidance_scale * (text_cond - text_uncond)
            else:
                text_pred = self.text_denoiser(
                    latents, vision_features, t_tensor, training=False
                )

            # Step
            latents, _ = self.scheduler.step(text_pred, t, latents)

        logger.info(f"Generated text embeddings via {num_inference_steps} diffusion steps")

        # Decode embeddings to tokens
        logits = self.text_decoder_head(latents)
        tokens = ops.argmax(logits, axis=-1)

        return tokens

    def compute_score_field(
            self,
            vision_features: keras.KerasTensor,
            text_features: keras.KerasTensor,
            timestep: int
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Compute the joint score field ∇ log p(image, text) at a point.

        This is the core of Protocol 3: treating the VLM as a unified world
        model whose score field can be queried at any (image, text) coordinate.

        Args:
            vision_features: Vision point [batch, vision_seq, vision_dim]
            text_features: Text point [batch, text_seq, text_dim]
            timestep: Noise level to query at

        Returns:
            Tuple of (vision_score, text_score) representing ∇ log p
        """
        if not hasattr(self, 'joint_denoiser'):
            raise ValueError("Model not configured for joint score computation")

        batch_size = ops.shape(vision_features)[0]
        t = ops.convert_to_tensor([timestep] * batch_size, dtype='int32')

        # Denoise to get predicted clean samples
        denoised_v, denoised_t = self.joint_denoiser(
            vision_features, text_features, t, training=False
        )

        # By Miyasawa: score = (denoised - noisy) / variance
        vision_score = self.scheduler.get_score_from_noise(
            denoised_v - vision_features, t, vision_features
        )
        text_score = self.scheduler.get_score_from_noise(
            denoised_t - text_features, t, text_features
        )

        return vision_score, text_score

    def navigate_semantic_space(
            self,
            start_vision: keras.KerasTensor,
            start_text: keras.KerasTensor,
            target_text: keras.KerasTensor,
            num_steps: int = 100,
            step_size: float = 0.01
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Navigate from one point to another in semantic space (Protocol 3).

        This implements "semantic calculus" - using the score field to
        traverse the manifold from one concept to another while staying in
        high-probability regions.

        Example: Given (image, "daytime"), navigate to (?, "nighttime")

        Args:
            start_vision: Starting vision features
            start_text: Starting text features
            target_text: Target text concept
            num_steps: Number of navigation steps
            step_size: Step size for gradient ascent

        Returns:
            Final (vision, text) coordinates after navigation
        """
        current_v = start_vision
        current_t = start_text

        # Compute direction in text space
        text_direction = target_text - start_text
        text_direction = text_direction / ops.norm(text_direction)

        # Navigate via gradient ascent on score field
        for step in range(num_steps):
            # Query score at current position
            timestep = self.scheduler.num_timesteps // 2  # Mid-noise level
            score_v, score_t = self.compute_score_field(current_v, current_t, timestep)

            # Move towards target while following score field
            # Text: move towards target
            current_t = current_t + step_size * text_direction
            # Vision: follow score (gradient ascent)
            current_v = current_v + step_size * score_v

            # Optional: project back onto data manifold periodically
            if step % 10 == 0:
                # Denoise slightly to stay on manifold
                t_denoise = ops.convert_to_tensor(
                    [timestep] * ops.shape(current_v)[0], dtype='int32'
                )
                current_v, current_t = self.joint_denoiser(
                    current_v, current_t, t_denoise, training=False
                )

        logger.info(f"Navigated semantic space in {num_steps} steps")
        return current_v, current_t

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config.update({
            'vision_config': self.vision_config,
            'text_config': self.text_config,
            'diffusion_config': self.diffusion_config,
            'vocab_size': self.vocab_size,
            'generation_mode': self.generation_mode,
            'use_classifier_free_guidance': self.use_classifier_free_guidance,
        })
        return config


# === Factory Functions ===

def create_score_based_nanovlm(
        variant: Literal['mini', 'base', 'large'] = 'base',
        mode: Literal['text_to_image', 'image_to_text', 'joint'] = 'joint',
        vocab_size: int = 32000,
        **kwargs
) -> ScoreBasedNanoVLM:
    """
    Create a score-based nanoVLM with predefined configurations.

    Args:
        variant: Model size ('mini', 'base', 'large')
        mode: Generation mode
        vocab_size: Vocabulary size
        **kwargs: Additional arguments

    Returns:
        Configured ScoreBasedNanoVLM
    """
    configs = {
        'mini': {
            'vision_config': {
                'img_size': 224, 'patch_size': 16, 'embed_dim': 384,
                'depth': 6, 'num_heads': 6
            },
            'text_config': {
                'vocab_size': vocab_size, 'embed_dim': 384,
                'depth': 6, 'num_heads': 6, 'max_seq_len': 512
            },
            'diffusion_config': {
                'num_timesteps': 1000, 'beta_schedule': 'cosine'
            }
        },
        'base': {
            'vision_config': {
                'img_size': 224, 'patch_size': 16, 'embed_dim': 768,
                'depth': 12, 'num_heads': 12
            },
            'text_config': {
                'vocab_size': vocab_size, 'embed_dim': 768,
                'depth': 12, 'num_heads': 12, 'max_seq_len': 512
            },
            'diffusion_config': {
                'num_timesteps': 1000, 'beta_schedule': 'cosine'
            }
        },
        'large': {
            'vision_config': {
                'img_size': 384, 'patch_size': 16, 'embed_dim': 1024,
                'depth': 24, 'num_heads': 16
            },
            'text_config': {
                'vocab_size': vocab_size, 'embed_dim': 1024,
                'depth': 24, 'num_heads': 16, 'max_seq_len': 1024
            },
            'diffusion_config': {
                'num_timesteps': 1000, 'beta_schedule': 'cosine'
            }
        }
    }

    config = configs[variant]

    return ScoreBasedNanoVLM(
        vision_config=config['vision_config'],
        text_config=config['text_config'],
        diffusion_config=config['diffusion_config'],
        vocab_size=vocab_size,
        generation_mode=mode,
        **kwargs
    )