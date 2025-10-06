"""
Denoiser Networks for Score-Based nanoVLM

Implements the core denoising networks that learn score functions via
Denoising Score Matching (DSM), following Miyasawa's theorem. These denoisers
are the foundation of the navigable world model.

References:
    - Vincent (2011): "A Connection Between Score Matching and Denoising Autoencoders"
    - Song & Ermon (2019): "Generative Modeling by Estimating Gradients"
    - Ho et al. (2020): "Denoising Diffusion Probabilistic Models"
"""

import keras
from keras import ops, layers
from typing import Optional, Dict, Any, Tuple

from dl_techniques.utils.logger import logger


@keras.saving.register_keras_serializable()
class TimestepEmbedding(layers.Layer):
    """
    Sinusoidal timestep embedding for diffusion models.

    Maps timestep indices to continuous embeddings using sinusoidal functions,
    similar to positional encoding in transformers. This allows the denoiser
    to condition on the noise level.

    Args:
        embedding_dim: Dimension of the timestep embedding. Should be even.
        max_period: Maximum period for sinusoidal embedding. Defaults to 10000.
        **kwargs: Additional layer arguments.
    """

    def __init__(
            self,
            embedding_dim: int,
            max_period: int = 10000,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)

        if embedding_dim % 2 != 0:
            raise ValueError(f"embedding_dim must be even, got {embedding_dim}")

        self.embedding_dim = embedding_dim
        self.max_period = max_period

        # Compute frequencies for sinusoidal embedding
        half_dim = embedding_dim // 2
        freqs = ops.exp(
            -ops.log(float(max_period)) *
            ops.arange(0, half_dim, dtype='float32') / half_dim
        )
        self.freqs = freqs

    def call(self, timesteps: keras.KerasTensor) -> keras.KerasTensor:
        """
        Compute sinusoidal timestep embeddings.

        Args:
            timesteps: Timestep indices of shape [batch] or [batch, 1]

        Returns:
            Timestep embeddings of shape [batch, embedding_dim]
        """
        # Ensure shape is [batch]
        if len(ops.shape(timesteps)) > 1:
            timesteps = ops.squeeze(timesteps, axis=-1)

        # Convert to float and expand dims
        timesteps = ops.cast(timesteps, 'float32')
        timesteps = ops.expand_dims(timesteps, -1)  # [batch, 1]

        # Compute arguments: timesteps * freqs
        args = timesteps * ops.expand_dims(self.freqs, 0)  # [batch, half_dim]

        # Apply sin and cos
        embedding_sin = ops.sin(args)
        embedding_cos = ops.cos(args)

        # Concatenate
        embedding = ops.concatenate([embedding_sin, embedding_cos], axis=-1)

        return embedding

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'embedding_dim': self.embedding_dim,
            'max_period': self.max_period,
        })
        return config


@keras.saving.register_keras_serializable()
class ConditionalDenoiser(layers.Layer):
    """
    Conditional denoiser network that learns score functions.

    This is the core component that implements Denoising Score Matching.
    By Miyasawa's theorem, the optimal denoiser D(x_t, c, t) provides the score:

        ∇_x log p(x_t | c) ≈ (1/σ²) * (D(x_t, c, t) - x_t)

    Args:
        data_dim: Dimension of the data to denoise (image or text embedding dim).
        condition_dim: Dimension of conditioning information.
        hidden_dim: Hidden dimension for processing. Defaults to 512.
        num_layers: Number of residual processing layers. Defaults to 6.
        dropout_rate: Dropout rate. Defaults to 0.1.
        use_self_attention: Whether to use self-attention layers. Defaults to True.
        num_attention_heads: Number of attention heads. Defaults to 8.
        **kwargs: Additional layer arguments.
    """

    def __init__(
            self,
            data_dim: int,
            condition_dim: int,
            hidden_dim: int = 512,
            num_layers: int = 6,
            dropout_rate: float = 0.1,
            use_self_attention: bool = True,
            num_attention_heads: int = 8,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.data_dim = data_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.use_self_attention = use_self_attention
        self.num_attention_heads = num_attention_heads

        # Timestep embedding
        self.time_embed = TimestepEmbedding(hidden_dim, name='time_embed')

        # Project timestep embedding to match data flow
        self.time_mlp = keras.Sequential([
            layers.Dense(hidden_dim * 4, activation='silu', name='time_mlp_1'),
            layers.Dense(hidden_dim, name='time_mlp_2')
        ], name='time_mlp')

        # Input projections
        self.data_proj = layers.Dense(hidden_dim, name='data_proj')
        self.condition_proj = layers.Dense(hidden_dim, name='condition_proj')

        # Processing blocks
        self.blocks = []
        for i in range(num_layers):
            block_layers = {
                'norm1': layers.LayerNormalization(name=f'block_{i}_norm1'),
                'dense1': layers.Dense(hidden_dim * 4, activation='gelu', name=f'block_{i}_dense1'),
                'dropout1': layers.Dropout(dropout_rate, name=f'block_{i}_dropout1'),
                'dense2': layers.Dense(hidden_dim, name=f'block_{i}_dense2'),
                'dropout2': layers.Dropout(dropout_rate, name=f'block_{i}_dropout2'),
            }

            if use_self_attention:
                block_layers['norm_attn'] = layers.LayerNormalization(name=f'block_{i}_norm_attn')
                block_layers['attention'] = layers.MultiHeadAttention(
                    num_heads=num_attention_heads,
                    key_dim=hidden_dim // num_attention_heads,
                    dropout=dropout_rate,
                    name=f'block_{i}_attention'
                )

            self.blocks.append(block_layers)

        # Output projection to data space
        self.output_proj = keras.Sequential([
            layers.LayerNormalization(name='output_norm'),
            layers.Dense(data_dim, kernel_initializer='zeros', name='output_proj')
        ], name='output_proj')

        logger.info(
            f"Initialized ConditionalDenoiser with {num_layers} layers, "
            f"hidden_dim={hidden_dim}, attention={use_self_attention}"
        )

    def call(
            self,
            noisy_data: keras.KerasTensor,
            condition: keras.KerasTensor,
            timesteps: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Denoise data conditioned on context and timestep.

        Args:
            noisy_data: Noisy data x_t of shape [batch, seq_len, data_dim]
            condition: Conditioning information c of shape [batch, cond_seq_len, condition_dim]
            timesteps: Timestep indices of shape [batch]
            training: Training mode flag

        Returns:
            Denoised data of shape [batch, seq_len, data_dim]
        """
        # Embed timestep
        t_emb = self.time_embed(timesteps)  # [batch, hidden_dim]
        t_emb = self.time_mlp(t_emb, training=training)  # [batch, hidden_dim]
        t_emb = ops.expand_dims(t_emb, 1)  # [batch, 1, hidden_dim]

        # Project inputs to hidden dimension
        x = self.data_proj(noisy_data)  # [batch, seq_len, hidden_dim]
        c = self.condition_proj(condition)  # [batch, cond_seq_len, hidden_dim]

        # Add timestep information via broadcasting
        x = x + t_emb

        # Concatenate data and condition for processing
        combined = ops.concatenate([x, c], axis=1)  # [batch, seq_len + cond_seq_len, hidden_dim]

        # Process through residual blocks
        h = combined
        for block in self.blocks:
            # Residual MLP block
            residual = h
            h = block['norm1'](h)
            h = block['dense1'](h)
            h = block['dropout1'](h, training=training)
            h = block['dense2'](h)
            h = block['dropout2'](h, training=training)
            h = h + residual

            # Optional self-attention
            if self.use_self_attention:
                residual = h
                h = block['norm_attn'](h)
                h = block['attention'](h, h, training=training)
                h = h + residual

        # Extract only the data portion (not condition)
        data_seq_len = ops.shape(noisy_data)[1]
        h_data = h[:, :data_seq_len, :]

        # Project back to data dimension
        denoised = self.output_proj(h_data)

        # Residual connection: output denoised = input + correction
        output = noisy_data + denoised

        return output

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'data_dim': self.data_dim,
            'condition_dim': self.condition_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout_rate': self.dropout_rate,
            'use_self_attention': self.use_self_attention,
            'num_attention_heads': self.num_attention_heads,
        })
        return config


@keras.saving.register_keras_serializable()
class VisionDenoiser(layers.Layer):
    """
    Denoiser for image data conditioned on text.

    Implements the text-to-image generation denoiser that learns:
        p(image | text) via Denoising Score Matching

    This follows Protocol 1 from the Miyasawa framework.

    Args:
        vision_config: Configuration for vision processing.
        text_dim: Dimension of text conditioning.
        num_layers: Number of denoising layers. Defaults to 12.
        **kwargs: Additional arguments.
    """

    def __init__(
            self,
            vision_config: Dict[str, Any],
            text_dim: int,
            num_layers: int = 12,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.vision_config = vision_config
        self.text_dim = text_dim
        self.num_layers = num_layers

        # Vision dimension from config
        self.vision_dim = vision_config.get('embed_dim', 768)

        # Conditional denoiser
        self.denoiser = ConditionalDenoiser(
            data_dim=self.vision_dim,
            condition_dim=text_dim,
            hidden_dim=self.vision_dim,
            num_layers=num_layers,
            name='vision_denoiser'
        )

        logger.info(f"Initialized VisionDenoiser for text-to-image generation")

    def call(
            self,
            noisy_vision: keras.KerasTensor,
            text_features: keras.KerasTensor,
            timesteps: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Denoise vision features conditioned on text.

        Args:
            noisy_vision: Noisy vision features [batch, vision_seq, vision_dim]
            text_features: Text conditioning [batch, text_seq, text_dim]
            timesteps: Diffusion timesteps [batch]
            training: Training flag

        Returns:
            Denoised vision features [batch, vision_seq, vision_dim]
        """
        return self.denoiser(noisy_vision, text_features, timesteps, training=training)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'vision_config': self.vision_config,
            'text_dim': self.text_dim,
            'num_layers': self.num_layers,
        })
        return config


@keras.saving.register_keras_serializable()
class TextDenoiser(layers.Layer):
    """
    Denoiser for text embeddings conditioned on images.

    Implements the image-to-text generation denoiser that learns:
        p(text | image) via Denoising Score Matching in embedding space

    This follows Protocol 2 from the Miyasawa framework - a radical departure
    from autoregressive decoding, instead doing holistic generation in latent space.

    Args:
        text_dim: Dimension of text embeddings.
        vision_dim: Dimension of vision conditioning.
        num_layers: Number of denoising layers. Defaults to 12.
        **kwargs: Additional arguments.
    """

    def __init__(
            self,
            text_dim: int,
            vision_dim: int,
            num_layers: int = 12,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.text_dim = text_dim
        self.vision_dim = vision_dim
        self.num_layers = num_layers

        # Conditional denoiser
        self.denoiser = ConditionalDenoiser(
            data_dim=text_dim,
            condition_dim=vision_dim,
            hidden_dim=max(text_dim, vision_dim),
            num_layers=num_layers,
            name='text_denoiser'
        )

        logger.info(f"Initialized TextDenoiser for image-to-text generation")

    def call(
            self,
            noisy_text: keras.KerasTensor,
            vision_features: keras.KerasTensor,
            timesteps: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Denoise text embeddings conditioned on vision.

        Args:
            noisy_text: Noisy text embeddings [batch, text_seq, text_dim]
            vision_features: Vision conditioning [batch, vision_seq, vision_dim]
            timesteps: Diffusion timesteps [batch]
            training: Training flag

        Returns:
            Denoised text embeddings [batch, text_seq, text_dim]
        """
        return self.denoiser(noisy_text, vision_features, timesteps, training=training)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'text_dim': self.text_dim,
            'vision_dim': self.vision_dim,
            'num_layers': self.num_layers,
        })
        return config


@keras.saving.register_keras_serializable()
class JointDenoiser(layers.Layer):
    """
    Joint denoiser for simultaneous vision and text denoising.

    Implements the unified denoiser that learns the joint score field:
        ∇ log p(image, text)

    This follows Protocol 3 from the Miyasawa framework - treating the VLM
    as a single unified world model where vision and language are different
    views of the same semantic landscape.

    Args:
        vision_dim: Vision feature dimension.
        text_dim: Text feature dimension.
        hidden_dim: Hidden processing dimension. Defaults to 1024.
        num_layers: Number of processing layers. Defaults to 16.
        **kwargs: Additional arguments.
    """

    def __init__(
            self,
            vision_dim: int,
            text_dim: int,
            hidden_dim: int = 1024,
            num_layers: int = 16,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Timestep embedding
        self.time_embed = TimestepEmbedding(hidden_dim, name='time_embed')

        # Separate projections for vision and text
        self.vision_proj = layers.Dense(hidden_dim, name='vision_proj')
        self.text_proj = layers.Dense(hidden_dim, name='text_proj')

        # Joint processing blocks with cross-attention
        self.joint_blocks = []
        for i in range(num_layers):
            block = {
                # Self-attention for each modality
                'vision_self_attn': layers.MultiHeadAttention(
                    num_heads=8, key_dim=hidden_dim // 8, name=f'block_{i}_vision_self_attn'
                ),
                'text_self_attn': layers.MultiHeadAttention(
                    num_heads=8, key_dim=hidden_dim // 8, name=f'block_{i}_text_self_attn'
                ),
                # Cross-attention between modalities
                'vision_cross_attn': layers.MultiHeadAttention(
                    num_heads=8, key_dim=hidden_dim // 8, name=f'block_{i}_vision_cross_attn'
                ),
                'text_cross_attn': layers.MultiHeadAttention(
                    num_heads=8, key_dim=hidden_dim // 8, name=f'block_{i}_text_cross_attn'
                ),
                # Norms and MLPs
                'vision_norm1': layers.LayerNormalization(name=f'block_{i}_vision_norm1'),
                'vision_norm2': layers.LayerNormalization(name=f'block_{i}_vision_norm2'),
                'vision_norm3': layers.LayerNormalization(name=f'block_{i}_vision_norm3'),
                'text_norm1': layers.LayerNormalization(name=f'block_{i}_text_norm1'),
                'text_norm2': layers.LayerNormalization(name=f'block_{i}_text_norm2'),
                'text_norm3': layers.LayerNormalization(name=f'block_{i}_text_norm3'),
                'vision_mlp': keras.Sequential([
                    layers.Dense(hidden_dim * 4, activation='gelu'),
                    layers.Dense(hidden_dim)
                ], name=f'block_{i}_vision_mlp'),
                'text_mlp': keras.Sequential([
                    layers.Dense(hidden_dim * 4, activation='gelu'),
                    layers.Dense(hidden_dim)
                ], name=f'block_{i}_text_mlp'),
            }
            self.joint_blocks.append(block)

        # Output projections
        self.vision_out = layers.Dense(vision_dim, kernel_initializer='zeros', name='vision_out')
        self.text_out = layers.Dense(text_dim, kernel_initializer='zeros', name='text_out')

        logger.info(f"Initialized JointDenoiser for unified vision-language score modeling")

    def call(
            self,
            noisy_vision: keras.KerasTensor,
            noisy_text: keras.KerasTensor,
            timesteps: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Jointly denoise vision and text features.

        Args:
            noisy_vision: Noisy vision [batch, vision_seq, vision_dim]
            noisy_text: Noisy text [batch, text_seq, text_dim]
            timesteps: Timesteps [batch]
            training: Training flag

        Returns:
            Tuple of (denoised_vision, denoised_text)
        """
        # Embed timestep
        t_emb = self.time_embed(timesteps)
        t_emb = ops.expand_dims(t_emb, 1)

        # Project to hidden dimension
        h_vision = self.vision_proj(noisy_vision) + t_emb
        h_text = self.text_proj(noisy_text) + t_emb

        # Process through joint blocks
        for block in self.joint_blocks:
            # Self-attention within each modality
            v_res = h_vision
            h_vision = block['vision_norm1'](h_vision)
            h_vision = block['vision_self_attn'](h_vision, h_vision, training=training)
            h_vision = h_vision + v_res

            t_res = h_text
            h_text = block['text_norm1'](h_text)
            h_text = block['text_self_attn'](h_text, h_text, training=training)
            h_text = h_text + t_res

            # Cross-attention between modalities
            v_res = h_vision
            h_vision = block['vision_norm2'](h_vision)
            h_vision = block['vision_cross_attn'](h_vision, h_text, training=training)
            h_vision = h_vision + v_res

            t_res = h_text
            h_text = block['text_norm2'](h_text)
            h_text = block['text_cross_attn'](h_text, h_vision, training=training)
            h_text = h_text + t_res

            # MLPs
            v_res = h_vision
            h_vision = block['vision_norm3'](h_vision)
            h_vision = block['vision_mlp'](h_vision, training=training)
            h_vision = h_vision + v_res

            t_res = h_text
            h_text = block['text_norm3'](h_text)
            h_text = block['text_mlp'](h_text, training=training)
            h_text = h_text + t_res

        # Project back to original dimensions with residual
        denoised_vision = noisy_vision + self.vision_out(h_vision)
        denoised_text = noisy_text + self.text_out(h_text)

        return denoised_vision, denoised_text

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'vision_dim': self.vision_dim,
            'text_dim': self.text_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
        })
        return config