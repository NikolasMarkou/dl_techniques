"""
FNet: Fourier Transform-based Attention Replacement

This module implements the FNet architecture from "FNet: Mixing Tokens with Fourier Transforms"
(Lee-Thorp et al., 2021), which replaces self-attention with parameter-free Fourier transforms
for efficient token mixing in transformer-style architectures.
"""

import keras
from typing import Optional

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.fnet_encoder_block import FNetEncoderBlock

# ---------------------------------------------------------------------

def create_fnet_encoder(
        num_layers: int,
        hidden_dim: int,
        max_seq_length: int,
        intermediate_dim: Optional[int] = None,
        dropout_rate: float = 0.1,
        **encoder_kwargs
) -> keras.Model:
    """
    Create a complete FNet encoder with multiple layers.

    Args:
        num_layers: Number of encoder layers. Must be positive.
        hidden_dim: Hidden dimension size. Must be positive.
        max_seq_length: The fixed sequence length for the model. Must be positive.
        intermediate_dim: Feed-forward intermediate dimension (defaults to 4*hidden_dim).
        dropout_rate: Dropout rate for all layers.
        **encoder_kwargs: Additional arguments for FNetEncoderBlock.

    Returns:
        Keras Model implementing multi-layer FNet encoder.
    """
    # --- Input Validation ---
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")
    if hidden_dim <= 0:
        raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
    if max_seq_length <= 0:
        raise ValueError(f"max_seq_length must be positive, got {max_seq_length}")

    if intermediate_dim is None:
        intermediate_dim = hidden_dim * 4

    inputs = keras.Input(shape=(max_seq_length, hidden_dim), name='encoder_input')
    x = inputs

    for i in range(num_layers):
        x = FNetEncoderBlock(
            intermediate_dim=intermediate_dim,
            dropout_rate=dropout_rate,
            name=f'fnet_encoder_layer_{i}',
            **encoder_kwargs
        )(x)

    return keras.Model(inputs, x, name=f'fnet_encoder_{num_layers}L_{hidden_dim}H')

# ---------------------------------------------------------------------

def create_fnet_classifier(
        num_classes: int,
        num_layers: int = 6,
        hidden_dim: int = 768,
        max_seq_length: int = 512,
        vocab_size: int = 30522,
        dropout_rate: float = 0.1,
        **kwargs
) -> keras.Model:
    """
    Create a complete FNet model for classification tasks.

    Args:
        num_classes: Number of output classes. Must be positive.
        num_layers: Number of encoder layers.
        hidden_dim: Hidden dimension size.
        max_seq_length: Maximum sequence length.
        vocab_size: Vocabulary size for embeddings. Must be positive.
        dropout_rate: Dropout rate throughout the model.
        **kwargs: Additional arguments for encoder blocks.

    Returns:
        Complete FNet model ready for classification training.
    """
    # --- Input Validation ---
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")

    # Input layers
    input_ids = keras.Input(shape=(max_seq_length,), dtype='int32', name='input_ids')

    # Embedding layers (with masking for padding if needed)
    token_embeddings = keras.layers.Embedding(
        vocab_size, hidden_dim, name='token_embeddings', mask_zero=True
    )(input_ids)

    position_embeddings = keras.layers.Embedding(
        max_seq_length, hidden_dim, name='position_embeddings'
    )(keras.ops.arange(max_seq_length))

    # Combine embeddings
    embeddings = token_embeddings + position_embeddings
    embeddings = keras.layers.LayerNormalization(name='embedding_layer_norm')(embeddings)
    embeddings = keras.layers.Dropout(dropout_rate, name='embedding_dropout')(embeddings)

    # FNet encoder
    encoded = create_fnet_encoder(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        max_seq_length=max_seq_length,  # Pass the fixed sequence length
        dropout_rate=dropout_rate,
        **kwargs
    )(embeddings)

    # Classification head
    pooled = encoded[:, 0, :]  # Use [CLS] token
    pooled = keras.layers.Dense(hidden_dim, activation='tanh', name='pooler')(pooled)
    outputs = keras.layers.Dense(num_classes, name='classifier')(pooled)

    return keras.Model(input_ids, outputs, name='fnet_classifier')

# ---------------------------------------------------------------------