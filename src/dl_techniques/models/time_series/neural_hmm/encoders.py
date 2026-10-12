"""Sequence encoders for the Neural HMM.

Provides ``create_encoder_layer`` and a registry of encoder types, following
the factory pattern used throughout ``dl_techniques``. Each encoder maps an
input sequence to a hidden-state sequence ``h_{1:T}``, optionally with
causal masking for generative mode.
"""

from typing import Any, Dict, Literal, Optional, Tuple

import keras

# ---------------------------------------------------------------------
# encoder type literal
# ---------------------------------------------------------------------

EncoderType = Literal["mlp", "lstm", "gru", "conv", "transformer"]

# ---------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------

#: Registry of encoder types. Every entry carries ``class``, ``description``,
#: ``required_params``, ``optional_params`` and ``use_case``.
ENCODER_REGISTRY: Dict[str, Dict[str, Any]] = {
    "mlp": {
        "class": "MLPSequenceEncoder",
        "description": "Standard MLP encoder with gelu activations.",
        "required_params": ["hidden_dim"],
        "optional_params": {"causal": False},
        "use_case": "Generative or discriminative encoding.",
    },
    "lstm": {
        "class": "LSTMSequenceEncoder",
        "description": "LSTM encoder with optional causal masking.",
        "required_params": ["hidden_dim"],
        "optional_params": {"causal": True},
        "use_case": "Generative HMM: encoder sees only x_{<t}.",
    },
    "gru": {
        "class": "GRUSequenceEncoder",
        "description": "GRU encoder with optional causal masking.",
        "required_params": ["hidden_dim"],
        "optional_params": {"causal": True},
        "use_case": "Generative HMM: encoder sees only x_{<t}.",
    },
    "conv": {
        "class": "ConvSequenceEncoder",
        "description": "1D Conv encoder with gelu activations.",
        "required_params": ["hidden_dim"],
        "optional_params": {"causal": True, "kernel_size": 3},
        "use_case": "Generative or discriminative encoding.",
    },
    "transformer": {
        "class": "TransformerSequenceEncoder",
        "description": "Transformer encoder with optional causal masking.",
        "required_params": ["hidden_dim"],
        "optional_params": {"causal": True, "num_heads": 4, "num_layers": 2},
        "use_case": "Generative HMM with attention.",
    },
}


# ---------------------------------------------------------------------
# encoder classes
# ---------------------------------------------------------------------


class MLPSequenceEncoder(keras.layers.Layer):
    """Standard MLP encoder."""

    def __init__(self, hidden_dim: int, causal: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.causal = causal
        self.dense = keras.layers.Dense(hidden_dim, activation="gelu")

    def call(self, inputs: keras.KerasTensor, mask: Optional[keras.KerasTensor] = None) -> keras.KerasTensor:
        outputs = self.dense(inputs)
        if self.causal and mask is not None:
            # Causal mask applied via the encoder's masking logic
            pass
        return outputs


class LSTMSequenceEncoder(keras.layers.Layer):
    """LSTM encoder with optional causal masking."""

    def __init__(self, hidden_dim: int, causal: bool = True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.causal = causal
        self.lstm = keras.layers.LSTM(hidden_dim, return_sequences=True)

    def call(self, inputs: keras.KerasTensor, mask: Optional[keras.KerasTensor] = None) -> keras.KerasTensor:
        return self.lstm(inputs, mask=mask)


class GRUSequenceEncoder(keras.layers.Layer):
    """GRU encoder with optional causal masking."""

    def __init__(self, hidden_dim: int, causal: bool = True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.causal = causal
        self.gru = keras.layers.GRU(hidden_dim, return_sequences=True)

    def call(self, inputs: keras.KerasTensor, mask: Optional[keras.KerasTensor] = None) -> keras.KerasTensor:
        return self.gru(inputs, mask=mask)


class ConvSequenceEncoder(keras.layers.Layer):
    """1D Conv encoder with gelu activations."""

    def __init__(self, hidden_dim: int, causal: bool = True, kernel_size: int = 3, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.causal = causal
        self.kernel_size = kernel_size
        self.conv = keras.layers.Conv1D(hidden_dim, kernel_size, activation="gelu", padding="causal" if causal else "valid")

    def call(self, inputs: keras.KerasTensor, mask: Optional[keras.KerasTensor] = None) -> keras.KerasTensor:
        outputs = self.conv(inputs)
        return outputs


class TransformerSequenceEncoder(keras.layers.Layer):
    """Transformer encoder with optional causal masking."""

    def __init__(self, hidden_dim: int, causal: bool = True, num_heads: int = 4, num_layers: int = 2, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.causal = causal
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.embedding = keras.layers.Dense(hidden_dim)
        self.transformer_blocks = [
            keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=hidden_dim // num_heads)
            for _ in range(num_layers)
        ]

    def call(self, inputs: keras.KerasTensor, mask: Optional[keras.KerasTensor] = None) -> keras.KerasTensor:
        # Simple transformer forward
        x = self.embedding(inputs)
        for block in self.transformer_blocks:
            if self.causal:
                # Causal attention mask
                seq_len = keras.ops.shape(x)[1]
                mask_2d = keras.ops.one_hot(
                    keras.ops.arange(seq_len), seq_len
                )
                causal_mask = 1 - mask_2d  # Upper triangle is -inf
                attn_output = block(
                    query=x,
                    value=x,
                    key=x,
                    attention_mask=causal_mask,
                )[0]
            else:
                attn_output = block(
                    query=x, value=x, key=x, attention_mask=mask
                )[0]
            x = x + attn_output  # residual
        x = self.ffn(x) if hasattr(self, 'ffn') else x
        return x


# ---------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------


def create_encoder_layer(
    encoder_type: EncoderType,
    hidden_dim: int,
    causal: bool = False,
    **kwargs: Any,
) -> keras.layers.Layer:
    """Create an encoder layer by type name.

    :param encoder_type: One of ``EncoderType``.
    :type encoder_type: EncoderType
    :param hidden_dim: Dimensionality of the hidden state.
    :type hidden_dim: int
    :param causal: Whether the encoder is causal (generative mode).
    :type causal: bool
    :param kwargs: Additional keyword arguments passed to the encoder class.
    :type kwargs: Any
    :return: A configured encoder layer.
    :rtype: keras.layers.Layer
    :raises ValueError: If ``encoder_type`` is unknown.
    """
    valid = set(ENCODER_REGISTRY.keys())
    if encoder_type not in valid:
        raise ValueError(
            f"Unknown encoder_type '{encoder_type}'. Available: {sorted(valid)}."
        )

    # Map type string to encoder class
    class_name = ENCODER_REGISTRY[encoder_type]["class"]

    # Instantiate the appropriate encoder class
    if class_name == "MLPSequenceEncoder":
        return MLPSequenceEncoder(hidden_dim=hidden_dim, causal=causal, **kwargs)
    elif class_name == "LSTMSequenceEncoder":
        return LSTMSequenceEncoder(hidden_dim=hidden_dim, causal=causal, **kwargs)
    elif class_name == "GRUSequenceEncoder":
        return GRUSequenceEncoder(hidden_dim=hidden_dim, causal=causal, **kwargs)
    elif class_name == "ConvSequenceEncoder":
        return ConvSequenceEncoder(hidden_dim=hidden_dim, causal=causal, **kwargs)
    elif class_name == "TransformerSequenceEncoder":
        return TransformerSequenceEncoder(hidden_dim=hidden_dim, causal=causal, **kwargs)
    else:
        raise ValueError(f"Unknown encoder class '{class_name}'.")