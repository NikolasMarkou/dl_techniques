"""
Inject positional information into a sequence using learnable embeddings.

This layer addresses the permutation-invariant nature of the Transformer
architecture. Core mechanisms like self-attention do not have a built-in
understanding of token order, treating input as an unordered set. This
layer explicitly encodes the position of each token by adding a unique,
trainable vector to its embedding.

Architecture:
    The fundamental design is an embedding lookup table where each row
    corresponds to an absolute position in the sequence. Unlike the fixed
    sinusoidal functions proposed in the original Transformer paper, this
    implementation uses *learnable* positional embeddings. This allows the
    model to learn the optimal representation of position for its specific
    downstream task, rather than being constrained to a predefined
    mathematical form.

    The process is as follows:
    1.  A weight matrix, representing the positional embedding table, is
        initialized with shape `(max_sequence_length, embedding_dimension)`.
    2.  For an incoming sequence of length `L`, the first `L` embedding
        vectors are sliced from this table.
    3.  These `L` positional vectors are added element-wise to the `L` token
        embedding vectors of the input sequence.

    This simple additive approach effectively merges semantic and positional
    information into a single, unified representation that can be processed
    by subsequent Transformer layers.

Foundational Mathematics:
    Let `X ∈ R^(L x D)` be the input tensor of token embeddings for a
    sequence of length `L` with dimension `D`. Let `P ∈ R^(M x D)` be the
    learnable positional embedding table, where `M` is the maximum possible
    sequence length.

    The output `Y ∈ R^(L x D)` is computed by adding the corresponding
    positional embedding to each token embedding:

        Y_i = X_i + P_i   for i = 0, 1, ..., L-1

    By projecting both token identity and position into the same vector
    space, the self-attention mechanism can learn to compute attention
    scores that are a function of both what a token is and where it is. For
    example, the dot product attention between two tokens `i` and `j` will
    depend on terms involving `X_i`, `X_j`, `P_i`, and `P_j`, allowing the
    model to learn relative positional relationships.

References:
    - The concept of adding positional encodings was introduced in the
      original Transformer paper, although it used a fixed sinusoidal
      function:
      Vaswani, A., et al. (2017). "Attention Is All You Need".

    - The use of *learnable* absolute positional embeddings, as implemented
      here, is a common variant used in highly influential models like BERT
      and GPT:
      Devlin, J., et al. (2018). "BERT: Pre-training of Deep Bidirectional
      Transformers for Language Understanding".
"""

import keras
from keras import ops
from typing import Optional, Dict, Any, Union, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class PositionalEmbedding(keras.layers.Layer):
    """Learned absolute positional embedding layer with dropout regularization.

    Adds learnable positional embeddings to input sequences, enabling the model
    to encode absolute token positions. A weight matrix ``P ∈ R^(M x D)`` is
    maintained as a lookup table; for an input sequence of length ``L``, the
    first ``L`` rows are sliced and added element-wise:
    ``Y_i = X_i + P_i  for i = 0, ..., L-1``. An optional dropout is applied
    after the addition for regularization.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────┐
        │  Input (batch, seq_len, dim)     │
        └───────────────┬──────────────────┘
                        ▼
        ┌──────────────────────────────────┐
        │  Slice positional table P[:L]    │
        │  P ∈ R^(1, max_seq_len, dim)     │
        └───────────────┬──────────────────┘
                        ▼
        ┌──────────────────────────────────┐
        │  Add: X + P[:L]                  │
        └───────────────┬──────────────────┘
                        ▼
        ┌──────────────────────────────────┐
        │  Dropout                         │
        └───────────────┬──────────────────┘
                        ▼
        ┌──────────────────────────────────┐
        │  Output (batch, seq_len, dim)    │
        └──────────────────────────────────┘

    :param max_seq_len: Maximum sequence length that this layer can handle.
        Must be positive.
    :type max_seq_len: int
    :param dim: Embedding dimension. Must match the last dimension of input
        and be positive.
    :type dim: int
    :param dropout_rate: Dropout rate applied after adding positional
        embeddings. Must be in ``[0, 1]``. Default is ``0.0``.
    :type dropout_rate: float
    :param pos_initializer: Initializer for positional embeddings. Default is
        ``"truncated_normal"``.
    :type pos_initializer: Union[str, keras.initializers.Initializer]
    :param scale: Standard deviation for truncated normal initialization when
        using default initializer. Must be positive. Default is ``0.02``.
    :type scale: float
    :param kwargs: Additional keyword arguments for the Layer base class.

    :raises ValueError: If ``max_seq_len``, ``dim``, or ``scale`` are not
        positive.
    :raises ValueError: If ``dropout_rate`` is not in ``[0, 1]``.
    :raises ValueError: If input dimension does not match expected ``dim``
        during build.
    """

    def __init__(
            self,
            max_seq_len: int,
            dim: int,
            dropout_rate: float = 0.0,
            pos_initializer: Union[str, keras.initializers.Initializer] = "truncated_normal",
            scale: float = 0.02,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout must be in [0, 1], got {dropout_rate}")
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")

        # Store ALL configuration parameters
        self.max_seq_len = max_seq_len
        self.dim = dim
        self.dropout_rate = dropout_rate
        self.scale = scale
        self.pos_initializer = keras.initializers.get(pos_initializer)

        # Handle default initializer configuration
        if isinstance(self.pos_initializer, keras.initializers.TruncatedNormal):
            # Override with custom scale if using default
            self.pos_initializer = keras.initializers.TruncatedNormal(stddev=self.scale)
        elif pos_initializer == "truncated_normal":
            # Use custom scale for string specification
            self.pos_initializer = keras.initializers.TruncatedNormal(stddev=self.scale)

        # CREATE sub-layer in __init__ (modern Keras 3 pattern)
        self.dropout = keras.layers.Dropout(self.dropout_rate, name="pos_dropout")

        # Weight will be initialized in build()
        self.pos_embedding = None

        logger.info(f"Initialized PositionalEmbedding with max_seq_len={self.max_seq_len}, "
                    f"dim={self.dim}, dropout={self.dropout_rate}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer by creating weights and building sub-layers.

        :param input_shape: Shape tuple of the input tensor
            ``(batch_size, seq_len, dim)``.
        :type input_shape: Tuple[Optional[int], ...]
        """
        # Validate input shape
        if len(input_shape) != 3:
            raise ValueError(
                f"Input must be 3D (batch_size, seq_len, dim), got shape {input_shape}"
            )

        if input_shape[-1] != self.dim:
            raise ValueError(
                f"Input dimension {input_shape[-1]} does not match expected dim {self.dim}"
            )

        # Create positional embeddings weight
        self.pos_embedding = self.add_weight(
            name="pos_embedding",
            shape=(1, self.max_seq_len, self.dim),
            initializer=self.pos_initializer,
            trainable=True,
        )

        # CRITICAL: Explicitly build sub-layers for robust serialization
        # Dropout doesn't change shape, so we can use input_shape
        self.dropout.build(input_shape)

        # Always call parent build at the end
        super().build(input_shape)

        logger.info(f"Built PositionalEmbedding with input_shape={input_shape}")

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Add positional embeddings to input tensor.

        :param inputs: Input tensor with shape ``(batch_size, seq_len, dim)``.
        :type inputs: keras.KerasTensor
        :param training: Whether the layer should behave in training mode or
            inference mode.
        :type training: Optional[bool]
        :return: Output tensor with positional embeddings added, same shape as
            input.
        :rtype: keras.KerasTensor
        """
        # Get sequence length using keras ops for backend compatibility
        input_shape = ops.shape(inputs)
        seq_len = input_shape[1]

        # Slice positional embeddings to match sequence length
        positions = ops.slice(
            self.pos_embedding,
            start_indices=(0, 0, 0),
            shape=(1, seq_len, self.dim)
        )

        # Add positional embeddings
        outputs = inputs + positions

        # Apply dropout
        outputs = self.dropout(outputs, training=training)

        return outputs

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple (same as input shape).
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        :return: Dictionary containing all ``__init__`` parameters for proper
            serialization.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "max_seq_len": self.max_seq_len,
            "dim": self.dim,
            "dropout_rate": self.dropout_rate,
            "pos_initializer": keras.initializers.serialize(self.pos_initializer),
            "scale": self.scale,
        })
        return config


# ---------------------------------------------------------------------
