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
from typing import List, Optional, Dict, Any, Tuple

from dl_techniques.utils.logger import logger


@keras.saving.register_keras_serializable()

def _principal_shape(input_shape: Any) -> Tuple[Optional[int], ...]:
    """Return the FIRST shape when handed a list of shapes, else the shape.

    # DECISION plan-2026-08-19T163559-499b6f0e/D-040
    The distinguishing test is whether the first ELEMENT is itself a sequence,
    NOT `isinstance(input_shape, (list, tuple))` -- a plain shape IS a tuple, so
    that test is true for both cases and returns the batch dimension. MEASURED
    on the three sites this replaces: `compute_output_shape((None, 4, 16))`
    returned **None** and `compute_output_shape((2, 4, 16))` returned **2**,
    while the two sites that were already correct (`TimestepEmbedding` and
    `JointDenoiser`) were untouched by this bug and are untouched by this fix.

    Interface contract (3 callers by design):
        :param input_shape: A single shape tuple, or a sequence of them.
        :returns: The single shape, or the first of the sequence.
        :raises: Nothing; an empty sequence returns itself.
    """
    if (isinstance(input_shape, (list, tuple)) and input_shape
            and isinstance(input_shape[0], (list, tuple))):
        return input_shape[0]
    return input_shape

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

    def compute_output_shape(self, input_shape):
        """Output shape: (batch,) -> (batch, embedding_dim)."""
        return (input_shape[0], self.embedding_dim)

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

        # Processing blocks.
        #
        # DECISION plan-2026-08-19T163559-499b6f0e/D-039
        # SEVEN FLAT, PARALLEL LISTS -- one per role, indexed by block -- and
        # NOT `self.blocks = [{'norm1': ..., 'dense1': ...}, ...]`. Do NOT
        # "tidy" them back into a list of dicts: Keras 3.8 does not write a
        # layer container nested two or more levels deep to
        # `model.weights.h5` when its owner is a `keras.layers.Layer`. It DOES
        # write the identical container when the owner is a `keras.Model`,
        # which is why the same shape is harmless in `models/accunet` and
        # `models/cliffordnet`.
        # MEASURED on this exact class before the change: a 44-weight
        # `ConditionalDenoiser` put 12 of 44 tensors and 2,848 of 9,408
        # parameters into the archive, 12 of 44 survived a
        # perturb / save / reload comparison, and the forward output moved by
        # 9.118214e-03 against a range of 2.552134e+00. At full model scale
        # that is 464 of 1,305 weight tensors never written. Note the
        # `build()` below already materialises every one of these layers, so
        # "overrides build()" does NOT protect against this -- the container
        # SHAPE is the property that matters.
        # Same family as the `ConvDecoder` fix in D-026.
        # See decisions.md D-039 and REF-9 in findings/audit-batch-6.md.
        self.block_norm1: List[layers.Layer] = []
        self.block_dense1: List[layers.Layer] = []
        self.block_dropout1: List[layers.Layer] = []
        self.block_dense2: List[layers.Layer] = []
        self.block_dropout2: List[layers.Layer] = []
        self.block_norm_attn: List[layers.Layer] = []
        self.block_attention: List[layers.Layer] = []

        for i in range(num_layers):
            self.block_norm1.append(
                layers.LayerNormalization(name=f'block_{i}_norm1'))
            self.block_dense1.append(layers.Dense(
                hidden_dim * 4, activation='gelu', name=f'block_{i}_dense1'))
            self.block_dropout1.append(
                layers.Dropout(dropout_rate, name=f'block_{i}_dropout1'))
            self.block_dense2.append(
                layers.Dense(hidden_dim, name=f'block_{i}_dense2'))
            self.block_dropout2.append(
                layers.Dropout(dropout_rate, name=f'block_{i}_dropout2'))

            # The two attention lists stay EMPTY when self-attention is off, so
            # no `None` ever enters a tracked container.
            if use_self_attention:
                self.block_norm_attn.append(
                    layers.LayerNormalization(name=f'block_{i}_norm_attn'))
                self.block_attention.append(layers.MultiHeadAttention(
                    num_heads=num_attention_heads,
                    key_dim=hidden_dim // num_attention_heads,
                    dropout=dropout_rate,
                    name=f'block_{i}_attention'
                ))

        # Output projection to data space
        self.output_proj = keras.Sequential([
            layers.LayerNormalization(name='output_norm'),
            layers.Dense(data_dim, kernel_initializer='zeros', name='output_proj')
        ], name='output_proj')

        logger.info(
            f"Initialized ConditionalDenoiser with {num_layers} layers, "
            f"hidden_dim={hidden_dim}, attention={use_self_attention}"
        )

    def build(self, input_shape: Any) -> None:
        """Explicitly build every sub-layer so a ``.keras`` reload restores all
        weights (M2).

        The ``MultiHeadAttention`` layers and the nested ``Sequential`` blocks
        do NOT survive a lazy first-call build across serialization: on reload
        the layer is unbuilt at weight-restore time, so its variables are
        silently re-initialized. Building them here (from stored config; only
        the feature dims matter, sequence dims stay dynamic) pins variable
        creation to the build phase so weight paths match on restore.

        Args:
            input_shape: Inbound shape(s); content is unused (all sub-layer
                weight shapes derive from the stored config).
        """
        hd = self.hidden_dim
        # Timestep MLP consumes the [B, hidden_dim] sinusoidal embedding.
        self.time_mlp.build((None, hd))
        # Input projections (last dim = data / condition feature dims).
        self.data_proj.build((None, None, self.data_dim))
        self.condition_proj.build((None, None, self.condition_dim))
        # Residual blocks all operate on hidden_dim sequences.
        block_shape = (None, None, hd)
        for i in range(self.num_layers):
            self.block_norm1[i].build(block_shape)
            self.block_dense1[i].build(block_shape)
            self.block_dense2[i].build((None, None, hd * 4))
            if self.use_self_attention:
                self.block_norm_attn[i].build(block_shape)
                self.block_attention[i].build(
                    query_shape=block_shape, value_shape=block_shape
                )
        self.output_proj.build(block_shape)
        super().build(input_shape)

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
        for i in range(self.num_layers):
            # Residual MLP block
            residual = h
            h = self.block_norm1[i](h)
            h = self.block_dense1[i](h)
            h = self.block_dropout1[i](h, training=training)
            h = self.block_dense2[i](h)
            h = self.block_dropout2[i](h, training=training)
            h = h + residual

            # Optional self-attention
            if self.use_self_attention:
                residual = h
                h = self.block_norm_attn[i](h)
                h = self.block_attention[i](h, h, training=training)
                h = h + residual

        # Extract only the data portion (not condition)
        data_seq_len = ops.shape(noisy_data)[1]
        h_data = h[:, :data_seq_len, :]

        # Project back to data dimension
        denoised = self.output_proj(h_data)

        # Residual connection: output denoised = input + correction
        output = noisy_data + denoised

        return output

    def compute_output_shape(self, input_shape):
        """Output shape matches noisy_data input shape."""
        return _principal_shape(input_shape)

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

    def build(self, input_shape: Any) -> None:
        """Build the inner ConditionalDenoiser so weights survive reload (M2)."""
        if not self.denoiser.built:
            self.denoiser.build(input_shape)
        super().build(input_shape)

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

    def compute_output_shape(self, input_shape):
        """Output shape matches noisy_vision input shape."""
        return _principal_shape(input_shape)

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

    def build(self, input_shape: Any) -> None:
        """Build the inner ConditionalDenoiser so weights survive reload (M2)."""
        if not self.denoiser.built:
            self.denoiser.build(input_shape)
        super().build(input_shape)

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

    def compute_output_shape(self, input_shape):
        """Output shape matches noisy_text input shape."""
        return _principal_shape(input_shape)

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

        # Joint processing blocks with cross-attention.
        #
        # DECISION plan-2026-08-19T163559-499b6f0e/D-039
        # TEN FLAT, PARALLEL LISTS, one per role. Same ruling and same reason as
        # `ConditionalDenoiser` above: a layer container nested >=2 deep owned
        # by a `keras.layers.Layer` is NOT written to `model.weights.h5`. Do not
        # restore the list-of-dicts. `build()` below materialises every one of
        # these layers and that does NOT help -- the container SHAPE is what
        # matters, not whether the layers are built.
        # See decisions.md D-039.
        self.block_vision_self_attn: List[layers.Layer] = []
        self.block_text_self_attn: List[layers.Layer] = []
        self.block_vision_cross_attn: List[layers.Layer] = []
        self.block_text_cross_attn: List[layers.Layer] = []
        self.block_vision_norm1: List[layers.Layer] = []
        self.block_vision_norm2: List[layers.Layer] = []
        self.block_vision_norm3: List[layers.Layer] = []
        self.block_text_norm1: List[layers.Layer] = []
        self.block_text_norm2: List[layers.Layer] = []
        self.block_text_norm3: List[layers.Layer] = []
        self.block_vision_mlp: List[layers.Layer] = []
        self.block_text_mlp: List[layers.Layer] = []

        for i in range(num_layers):
            # Self-attention for each modality
            self.block_vision_self_attn.append(layers.MultiHeadAttention(
                num_heads=8, key_dim=hidden_dim // 8,
                name=f'block_{i}_vision_self_attn'))
            self.block_text_self_attn.append(layers.MultiHeadAttention(
                num_heads=8, key_dim=hidden_dim // 8,
                name=f'block_{i}_text_self_attn'))
            # Cross-attention between modalities
            self.block_vision_cross_attn.append(layers.MultiHeadAttention(
                num_heads=8, key_dim=hidden_dim // 8,
                name=f'block_{i}_vision_cross_attn'))
            self.block_text_cross_attn.append(layers.MultiHeadAttention(
                num_heads=8, key_dim=hidden_dim // 8,
                name=f'block_{i}_text_cross_attn'))
            # Norms and MLPs
            self.block_vision_norm1.append(
                layers.LayerNormalization(name=f'block_{i}_vision_norm1'))
            self.block_vision_norm2.append(
                layers.LayerNormalization(name=f'block_{i}_vision_norm2'))
            self.block_vision_norm3.append(
                layers.LayerNormalization(name=f'block_{i}_vision_norm3'))
            self.block_text_norm1.append(
                layers.LayerNormalization(name=f'block_{i}_text_norm1'))
            self.block_text_norm2.append(
                layers.LayerNormalization(name=f'block_{i}_text_norm2'))
            self.block_text_norm3.append(
                layers.LayerNormalization(name=f'block_{i}_text_norm3'))
            self.block_vision_mlp.append(keras.Sequential([
                layers.Dense(hidden_dim * 4, activation='gelu'),
                layers.Dense(hidden_dim)
            ], name=f'block_{i}_vision_mlp'))
            self.block_text_mlp.append(keras.Sequential([
                layers.Dense(hidden_dim * 4, activation='gelu'),
                layers.Dense(hidden_dim)
            ], name=f'block_{i}_text_mlp'))

        # Output projections
        self.vision_out = layers.Dense(vision_dim, kernel_initializer='zeros', name='vision_out')
        self.text_out = layers.Dense(text_dim, kernel_initializer='zeros', name='text_out')

        logger.info(f"Initialized JointDenoiser for unified vision-language score modeling")

    def build(self, input_shape: Any) -> None:
        """Explicitly build every sub-layer so a ``.keras`` reload restores all
        weights (M2).

        The 4 ``MultiHeadAttention`` layers per block and the per-block
        ``Sequential`` MLPs are lazily built otherwise and silently drop their
        weights on reload. All weight shapes derive from the stored config
        (sequence dims stay dynamic).

        Args:
            input_shape: Inbound shape(s); content is unused.
        """
        hd = self.hidden_dim
        self.vision_proj.build((None, None, self.vision_dim))
        self.text_proj.build((None, None, self.text_dim))
        block_shape = (None, None, hd)
        for i in range(self.num_layers):
            for attention in (
                self.block_vision_self_attn[i], self.block_text_self_attn[i],
                self.block_vision_cross_attn[i], self.block_text_cross_attn[i],
            ):
                attention.build(
                    query_shape=block_shape, value_shape=block_shape
                )
            for norm in (
                self.block_vision_norm1[i], self.block_vision_norm2[i],
                self.block_vision_norm3[i], self.block_text_norm1[i],
                self.block_text_norm2[i], self.block_text_norm3[i],
            ):
                norm.build(block_shape)
            self.block_vision_mlp[i].build(block_shape)
            self.block_text_mlp[i].build(block_shape)
        self.vision_out.build(block_shape)
        self.text_out.build(block_shape)
        super().build(input_shape)

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
        for i in range(self.num_layers):
            # Self-attention within each modality
            v_res = h_vision
            h_vision = self.block_vision_norm1[i](h_vision)
            h_vision = self.block_vision_self_attn[i](h_vision, h_vision, training=training)
            h_vision = h_vision + v_res

            t_res = h_text
            h_text = self.block_text_norm1[i](h_text)
            h_text = self.block_text_self_attn[i](h_text, h_text, training=training)
            h_text = h_text + t_res

            # Cross-attention between modalities
            v_res = h_vision
            h_vision = self.block_vision_norm2[i](h_vision)
            h_vision = self.block_vision_cross_attn[i](h_vision, h_text, training=training)
            h_vision = h_vision + v_res

            t_res = h_text
            h_text = self.block_text_norm2[i](h_text)
            h_text = self.block_text_cross_attn[i](h_text, h_vision, training=training)
            h_text = h_text + t_res

            # MLPs
            v_res = h_vision
            h_vision = self.block_vision_norm3[i](h_vision)
            h_vision = self.block_vision_mlp[i](h_vision, training=training)
            h_vision = h_vision + v_res

            t_res = h_text
            h_text = self.block_text_norm3[i](h_text)
            h_text = self.block_text_mlp[i](h_text, training=training)
            h_text = h_text + t_res

        # Project back to original dimensions with residual
        denoised_vision = noisy_vision + self.vision_out(h_vision)
        denoised_text = noisy_text + self.text_out(h_text)

        return denoised_vision, denoised_text

    def compute_output_shape(self, input_shape):
        """Returns tuple of (vision_shape, text_shape) matching inputs."""
        return (input_shape[0], input_shape[1])

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'vision_dim': self.vision_dim,
            'text_dim': self.text_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
        })
        return config