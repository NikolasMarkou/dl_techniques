"""A configurable Transformer text decoder, built by :class:`TextDecoder`.

The layer follows the decoder side of the Transformer: causal self-attention
restricts each position's representation to tokens at or before it, so the
stack models P(x_i | x_1, ..., x_{i-1}) and is suited to text generation.
Embedding type, positional encoding, attention type, normalization type and
position, and FFN type are all constructor arguments routed through factory
components, covering architectures from the original GPT decoder to a
modern RMSNorm + SwiGLU stack. Causal masking is applied automatically inside
`call()`; an optional padding mask is combined with it.

References:
    - Vaswani et al., 2017. Attention Is All You Need.
    - Radford et al., 2018. Improving Language Understanding by Generative
      Pre-Training.
    - Zhang & Sennrich, 2019. Root Mean Square Layer Normalization.
    - Shazeer, 2020. GLU Variants Improve Transformer.
"""

import math
import keras
from keras import ops, layers, initializers
from typing import Optional, Dict, Any, Literal, Tuple, Union, Callable

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.masking import create_mask, MaskConfig, combine_masks
from ..embedding import create_embedding_layer
from ..norms import create_normalization_layer, NormalizationType
from .transformer import TransformerLayer, AttentionType, FFNType, NormalizationPositionType
from dl_techniques.utils.activation_serialization import (
    serialize_activation,
    deserialize_activation,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------

PositionalType = Literal['learned', 'sincos']
# F-11: 'shared' is NOT a member -- it was dead code (an alias of 'learned' with
# no tying mechanism) and is now rejected in __init__. See the raise there.
EmbeddingType = Literal['learned', 'factorized']

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.transformers.text_decoder")
class TextDecoder(keras.layers.Layer):
    """
    General-purpose configurable text decoder built on a TransformerLayer stack.

    Orchestrates token and positional embeddings, causal masking, a stack of
    configurable transformer decoder blocks, and final normalization for
    autoregressive text generation. Causal masking is applied automatically;
    an optional padding mask can be combined with it.

    Architecture:

    .. code-block:: text

        ┌──────────────────────────────────────────┐
        │  Input IDs (B, seq_len)                  │
        └──────────────────┬───────────────────────┘
                           ▼
        ┌──────────────────────────────────────────┐
        │  Word Embedding + Positional Embedding   │
        │  ─► Embed Norm ─► Embed Dropout          │
        └──────────────────┬───────────────────────┘
                           ▼
        ┌──────────────────────────────────────────┐
        │  Causal + Padding Mask                   │
        └──────────────────┬───────────────────────┘
                           ▼
        ┌──────────────────────────────────────────┐
        │  TransformerLayer x depth                │
        │  (causal self-attention + FFN)           │
        └──────────────────┬───────────────────────┘
                           ▼
        ┌──────────────────────────────────────────┐
        │  Final Normalization                     │
        └──────────────────┬───────────────────────┘
                           ▼
        ┌──────────────────────────────────────────┐
        │  Output (B, seq_len, embed_dim)          │
        └──────────────────────────────────────────┘

    :param vocab_size: Vocabulary size.
    :type vocab_size: int
    :param embed_dim: Token embedding / hidden dimension.
    :type embed_dim: int
    :param depth: Number of decoder layers.
    :type depth: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param max_seq_len: Maximum sequence length. Default: 512.
    :type max_seq_len: int
    :param embedding_type: Word embedding strategy, ``'learned'`` or
        ``'factorized'``. Default: ``'learned'``. ``'shared'`` (a tied
        input/output embedding) is REJECTED: this layer has no output
        projection to tie to -- do the tying in the model that owns one.
    :type embedding_type: EmbeddingType
    :param positional_type: Positional encoding strategy. Default: ``'learned'``.
    :type positional_type: PositionalType
    :param attention_type: Attention mechanism type. Default: ``'multi_head'``.
    :type attention_type: AttentionType
    :param normalization_type: Normalization type. Default: ``'layer_norm'``.
    :type normalization_type: NormalizationType
    :param normalization_position: ``'pre'`` or ``'post'``. Default: ``'post'``.
    :type normalization_position: NormalizationPositionType
    :param ffn_type: FFN architecture type. Default: ``'mlp'``.
    :type ffn_type: FFNType
    :param stochastic_depth_rate: Drop-path rate. Default: 0.0.
    :type stochastic_depth_rate: float
    :param dropout_rate: Dropout rate. Default: 0.1.
    :type dropout_rate: float
    :param attention_dropout_rate: Attention dropout. Default: 0.1.
    :type attention_dropout_rate: float
    :param initializer_range: Std-dev for TruncatedNormal. Default: 0.02.
    :type initializer_range: float
    :param scale_residual_initializer_by_depth: When True, the two residual-path
        output projections of every block (the attention output projection and
        the FFN's contracting projection) are initialized at
        ``initializer_range / sqrt(2 * depth)`` instead of ``initializer_range``,
        so the residual stream's variance does not grow with depth. Q/K/V
        and the FFN expansion are unaffected. Default: False. This is GPT-2's
        published rule; see ``models/language/gpt2/gpt2.py`` and
        ``huggingface/transformers`` ``modeling_gpt2.py::_init_weights``.
        Requires ``attention_type='multi_head'`` and ``ffn_type='mlp'``; any
        other choice raises from the respective layer factory rather than
        silently ignoring the request.
    :type scale_residual_initializer_by_depth: bool
    :param layer_norm_eps: Normalization epsilon. Default: 1e-12. Applies to
        ``embed_norm``, ``final_norm``, and every one of the ``2 * depth``
        in-block norms (see decisions.md D-007 for why the in-block norms
        need this stated explicitly).
    :type layer_norm_eps: float
    :param kwargs: Additional keyword arguments for the base Layer.
    :type kwargs: Any

    :raises ValueError: If dimension parameters are invalid, or if
        ``embedding_type`` is not one of ``'learned'`` / ``'factorized'``.
    """

    def __init__(
            self,
            vocab_size: int,
            embed_dim: int,
            depth: int,
            num_heads: int,
            max_seq_len: int = 512,
            embedding_type: EmbeddingType = 'learned',
            positional_type: PositionalType = 'learned',
            attention_type: AttentionType = 'multi_head',
            normalization_type: NormalizationType = 'layer_norm',
            normalization_position: NormalizationPositionType = 'post',
            ffn_type: FFNType = 'mlp',
            activation: Union[str, Callable] = 'gelu',
            stochastic_depth_rate: float = 0.0,
            dropout_rate: float = 0.1,
            attention_dropout_rate: float = 0.1,
            initializer_range: float = 0.02,
            scale_residual_initializer_by_depth: bool = False,
            layer_norm_eps: float = 1e-12,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # --- Parameter Validation ---
        if not all(isinstance(p, int) and p > 0 for p in [vocab_size, embed_dim, depth, num_heads, max_seq_len]):
            raise ValueError("All dimension and size parameters must be positive integers.")
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}).")
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(f"dropout must be between 0.0 and 1.0, got {dropout_rate}")
        if not 0.0 <= attention_dropout_rate <= 1.0:
            raise ValueError(f"attention_dropout must be between 0.0 and 1.0, got {attention_dropout_rate}")
        if not 0.0 <= stochastic_depth_rate <= 1.0:
            raise ValueError(f"stochastic_depth_rate must be between 0.0 and 1.0, got {stochastic_depth_rate}")
        # Fail loud on an unknown embedding_type -- otherwise call() raises an
        # opaque AttributeError at first forward pass instead.
        # DECISION plan-2026-07-31T132403-b3f540cb/D-017: reject 'shared', don't
        # implement tying here -- this class has no output projection to tie to.
        # Tying belongs in the model that owns the vocabulary projection (see decisions.md).
        if embedding_type == 'shared':
            raise ValueError(
                "embedding_type='shared' is not supported: TextDecoder has no "
                "output/vocabulary projection to tie the word embedding to, so "
                "there is nothing to share with. Use 'learned' here and perform "
                "the tying in the model that owns the output projection (see "
                "models/language/masked_language_model/clm.py's tie_weights). "
                "Legal values: 'learned', 'factorized'."
            )
        if embedding_type not in ('learned', 'factorized'):
            raise ValueError(
                f"embedding_type must be one of 'learned', 'factorized', "
                f"got {embedding_type!r}"
            )

        # --- Store Configuration ---
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.embedding_type = embedding_type
        self.positional_type = positional_type
        self.attention_type = attention_type
        self.normalization_type = normalization_type
        self.normalization_position = normalization_position
        self.ffn_type = ffn_type
        self.activation = deserialize_activation(activation)
        self.stochastic_depth_rate = stochastic_depth_rate
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.initializer_range = initializer_range
        self.scale_residual_initializer_by_depth = bool(scale_residual_initializer_by_depth)
        self.layer_norm_eps = layer_norm_eps

        # --- Create Sub-layers in __init__ ---
        self._create_word_embeddings()
        self._create_positional_embeddings()

        # Embedding processing layers
        self.embed_dropout_layer = layers.Dropout(rate=self.dropout_rate, name="embed_dropout")
        self.embed_norm = create_normalization_layer(
            self.normalization_type, epsilon=self.layer_norm_eps, name="embed_norm"
        )

        # Create transformer decoder layers
        # DECISION plan-2026-08-18T140459-7991552f/D-067: kernel_initializer
        # below must not be dropped -- without it every block silently fell back to TransformerLayer's glorot_uniform instead of initializer_range. See decisions.md.
        # DECISION plan-2026-08-22T035419-a11304c8/D-160: use 1/sqrt(2 * depth),
        # not 1/sqrt(depth) -- the 2 counts residual additions per block (attention + FFN), matching upstream GPT-2's _init_weights. See decisions.md.
        residual_output_kernel_initializer = None
        if self.scale_residual_initializer_by_depth:
            residual_output_kernel_initializer = initializers.TruncatedNormal(
                stddev=self.initializer_range / math.sqrt(2.0 * self.depth)
            )

        self.decoder_layers = []
        for i in range(self.depth):
            # Linearly increase drop rate per layer
            layer_drop_rate = self.stochastic_depth_rate * i / max(1, self.depth - 1)
            # DECISION plan-2026-08-19T070627-a616f581/D-007: pass
            # attention_norm_args/ffn_norm_args so block norms track layer_norm_eps -- they used to fall back to the factory's 1e-6 default while final_norm ran at 1e-12. See decisions.md.
            layer = TransformerLayer(
                hidden_size=self.embed_dim,
                num_heads=self.num_heads,
                intermediate_size=int(self.embed_dim * 4),  # Standard 4x expansion
                attention_type=self.attention_type,
                normalization_type=self.normalization_type,
                attention_norm_args={'epsilon': self.layer_norm_eps},
                ffn_norm_args={'epsilon': self.layer_norm_eps},
                normalization_position=self.normalization_position,
                ffn_type=self.ffn_type,
                activation=self.activation,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                use_stochastic_depth=self.stochastic_depth_rate > 0.0,
                stochastic_depth_rate=layer_drop_rate,
                kernel_initializer=initializers.TruncatedNormal(
                    stddev=self.initializer_range
                ),
                residual_output_kernel_initializer=residual_output_kernel_initializer,
                name=f"decoder_layer_{i}"
            )
            self.decoder_layers.append(layer)

        # Final normalization layer
        self.final_norm = create_normalization_layer(
            self.normalization_type, epsilon=self.layer_norm_eps, name='final_norm'
        )

    def _create_word_embeddings(self) -> None:
        """Create word embedding layers based on the specified strategy."""
        initializer = initializers.TruncatedNormal(stddev=self.initializer_range)

        if self.embedding_type == 'learned':
            self.word_embeddings = layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embed_dim,
                embeddings_initializer=initializer,
                name="word_embeddings"
            )
        elif self.embedding_type == 'factorized':
            # Use factorized embeddings for memory efficiency
            factorized_dim = min(self.embed_dim, 128)
            self.factorized_embed_layer = layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=factorized_dim,
                embeddings_initializer=initializer,
                name='factorized_embed'
            )
            self.embed_projection_layer = layers.Dense(
                units=self.embed_dim,
                use_bias=False,
                kernel_initializer=initializer,
                name='embed_projection'
            )

    def _create_positional_embeddings(self) -> None:
        """Create positional embedding layer based on the specified strategy."""
        if self.positional_type == 'learned':
            self.positional_embeddings = layers.Embedding(
                input_dim=self.max_seq_len,
                output_dim=self.embed_dim,
                embeddings_initializer=initializers.TruncatedNormal(stddev=self.initializer_range),
                name="positional_embeddings"
            )
        elif self.positional_type == 'sincos':
            self.positional_embeddings = create_embedding_layer(
                'continuous_sincos',
                dim=self.embed_dim,
                ndim=1,
                name="positional_embeddings"
            )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all sub-layers.

        Builds each sub-layer explicitly so every weight variable exists
        before weight restoration during loading.

        :param input_shape: Input shape ``(batch, seq_len)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the STATIC sequence length exceeds ``max_seq_len``.
        """
        # Build word embedding layers
        if self.built:
            return

        # DECISION plan-2026-07-31T132403-b3f540cb/D-018: static seq_len guard
        # stays here, not in call() or a lower-level helper. Cannot fire on a dynamic sequence axis or on a later call to an already-built layer -- callers must bound seq_len themselves. See decisions.md.
        if input_shape is not None and len(input_shape) >= 2:
            static_seq_len = input_shape[1]
            if static_seq_len is not None and static_seq_len > self.max_seq_len:
                raise ValueError(
                    f"seq_len={static_seq_len} exceeds max_seq_len={self.max_seq_len}. "
                    f"This TextDecoder was configured with max_seq_len="
                    f"{self.max_seq_len}, so its positional encoding covers only "
                    f"{self.max_seq_len} positions and positions "
                    f"{self.max_seq_len}..{static_seq_len - 1} have no defined "
                    f"encoding. Construct the layer with max_seq_len >= "
                    f"{static_seq_len}, or truncate the input to at most "
                    f"{self.max_seq_len} tokens."
                )

        if hasattr(self, 'word_embeddings'):
            self.word_embeddings.build(input_shape)
        elif hasattr(self, 'factorized_embed_layer'):
            self.factorized_embed_layer.build(input_shape)
            # Compute shape after factorized embedding
            factorized_output_shape = self.factorized_embed_layer.compute_output_shape(input_shape)
            self.embed_projection_layer.build(factorized_output_shape)

        # Build positional embeddings with appropriate input shapes
        if self.positional_type == 'learned':
            # Learned embeddings take position indices as input
            position_input_shape = (None,)  # 1D sequence of positions
            self.positional_embeddings.build(position_input_shape)
        elif self.positional_type == 'sincos':
            # Continuous sincos embeddings take coordinates as input
            sincos_input_shape = (input_shape[0], input_shape[1], 1)  # (batch, seq, coord_dim)
            self.positional_embeddings.build(sincos_input_shape)

        # Compute embedding output shape for subsequent layers
        embedding_output_shape = (*input_shape, self.embed_dim)

        # Build embedding processing layers
        self.embed_norm.build(embedding_output_shape)
        self.embed_dropout_layer.build(embedding_output_shape)

        # Build all transformer decoder layers
        for layer in self.decoder_layers:
            layer.build(embedding_output_shape)

        # Build final normalization
        self.final_norm.build(embedding_output_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            input_ids: keras.KerasTensor,
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the text decoder.

        :param input_ids: Token IDs ``(B, seq_len)``.
        :type input_ids: keras.KerasTensor
        :param attention_mask: Optional padding mask ``(B, seq_len)``.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Training mode flag.
        :type training: Optional[bool]
        :return: Contextualized representations ``(B, seq_len, embed_dim)``.
        :rtype: keras.KerasTensor
        """
        seq_len = ops.shape(input_ids)[1]
        batch_size = ops.shape(input_ids)[0]

        # 1. Word Embeddings
        if self.embedding_type == 'factorized':
            x = self.factorized_embed_layer(input_ids)
            x = self.embed_projection_layer(x)
        else:
            x = self.word_embeddings(input_ids)

        # 2. Positional Embeddings
        positions = ops.arange(start=0, stop=seq_len)
        if self.positional_type == 'learned':
            pos_embed = self.positional_embeddings(positions)
            x = ops.add(x, pos_embed)
        elif self.positional_type == 'sincos':
            # Cast to this layer's compute dtype; measured bit-identical under
            # float32. Does not fix sincos under mixed_float16/float64 -- that cast lives inside continuous_sin_cos_embedding.py.
            pos_coords = ops.cast(positions, self.compute_dtype)
            pos_coords = ops.expand_dims(pos_coords, axis=-1)
            pos_coords = ops.expand_dims(pos_coords, axis=0)
            pos_coords = ops.broadcast_to(pos_coords, (batch_size, seq_len, 1))
            pos_embed = self.positional_embeddings(pos_coords)
            x = ops.add(x, pos_embed)

        # 3. Embedding normalization and dropout
        x = self.embed_norm(x, training=training)
        x = self.embed_dropout_layer(x, training=training)

        # 4. Attention mask: build in block-semantics (True = mask out), then
        # invert once at the end for the attention layer's attend-semantics (True = allow).

        # Create causal mask (True = future position to block)
        causal_mask = create_mask('causal', seq_len=seq_len, dtype='bool')
        # Add batch dimension and broadcast
        causal_mask = ops.expand_dims(causal_mask, axis=0)
        causal_mask = ops.broadcast_to(causal_mask, (batch_size, seq_len, seq_len))

        if attention_mask is not None:
            # Convert attention_mask from 1/0 format to boolean padding mask
            # True indicates padding (positions to block)
            padding_mask_1d = ops.equal(attention_mask, 0)  # Shape: (batch, seq_len)

            # Create padding attention mask using the factory
            padding_config = MaskConfig(
                mask_type='padding',
                dtype='bool',
                extra_params={'padding_mask': padding_mask_1d}
            )
            padding_mask_3d = create_mask(config=padding_config)  # Shape: (batch, seq_len, seq_len)

            # Combine causal and padding masks (True = block in either case)
            combined_mask = combine_masks(causal_mask, padding_mask_3d, combination='or')
        else:
            combined_mask = causal_mask

        # Invert: convert from block-semantics (True=block) to
        # attend-semantics (True=attend) expected by the attention layer
        attend_mask = ops.logical_not(combined_mask)

        # 5. Apply Transformer Layers
        for layer in self.decoder_layers:
            x = layer(x, attention_mask=attend_mask, training=training)

        # 6. Final Normalization
        x = self.final_norm(x, training=training)
        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape given input shape."""
        return (*input_shape, self.embed_dim)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary for serialization."""
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'depth': self.depth,
            'num_heads': self.num_heads,
            'max_seq_len': self.max_seq_len,
            'embedding_type': self.embedding_type,
            'positional_type': self.positional_type,
            'attention_type': self.attention_type,
            'normalization_type': self.normalization_type,
            'normalization_position': self.normalization_position,
            'ffn_type': self.ffn_type,
            'activation': serialize_activation(self.activation),
            'stochastic_depth_rate': self.stochastic_depth_rate,
            'dropout_rate': self.dropout_rate,
            'attention_dropout_rate': self.attention_dropout_rate,
            'initializer_range': self.initializer_range,
            'scale_residual_initializer_by_depth': self.scale_residual_initializer_by_depth,
            'layer_norm_eps': self.layer_norm_eps,
        })
        return config

# ---------------------------------------------------------------------
