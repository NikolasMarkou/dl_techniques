"""
This module provides a `PositionalEmbedding` layer, a fundamental component for
sequence-processing models like Transformers, implemented in Keras.

In many sequence models, especially Transformers, the core mechanisms (like self-attention)
are inherently order-agnostic; they treat the input as a "bag of tokens" without any
innate sense of position. This layer addresses this by adding a unique, trainable
vector to each position in an input sequence. This allows the model to learn the
significance of word order and relative positioning directly from the data.

Key Features and Mechanisms:

1.  **Learnable Embeddings:**
    Unlike fixed sinusoidal embeddings, the positional vectors in this layer are
    trainable weights. This gives the model the flexibility to discover optimal
    positional representations for a specific task and dataset, rather than relying
    on a predefined mathematical function.

2.  **Handling Variable Sequence Lengths:**
    The layer is designed to handle input sequences of variable lengths up to the
    configured `max_seq_len`. During the forward pass, it dynamically slices its
    internal embedding table to match the length of the incoming sequence before
    adding the positional vectors. This makes it efficient and flexible for use
    with batched data where sequences may have different lengths.

The operational flow of the layer is as follows:
-   An internal embedding table of shape `(max_seq_len, dim)` is created and learned.
-   For a given input tensor of shape `(batch_size, seq_len, dim)`, the layer takes
    the first `seq_len` vectors from its internal table.
-   These positional vectors are broadcasted and added to the input tensor.
-   Dropout is applied to the resulting tensor.
"""

import keras
from keras import ops
from keras import layers
from typing import Optional, Dict, Any, Union, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class PositionalEmbedding(keras.layers.Layer):
    """Learned positional embedding layer with enhanced stability.

    This layer adds learnable positional embeddings to input sequences, allowing
    the model to understand the position of elements in the sequence. The embeddings
    are learned during training and can capture complex positional relationships.

    This layer follows the modern Keras 3 pattern where sub-layers are created in
    __init__() and weights are created in build() for robust serialization.

    Args:
        max_seq_len: Integer, maximum sequence length that this layer can handle.
            Must be positive.
        dim: Integer, embedding dimension. Must match the last dimension of input
            and be positive.
        dropout: Float, dropout rate applied after adding positional embeddings.
            Must be in [0, 1]. Default is 0.0 (no dropout).
        pos_initializer: String or Initializer, initializer for positional embeddings.
            Default is "truncated_normal".
        scale: Float, standard deviation for truncated normal initialization when
            using default initializer. Must be positive. Default is 0.02.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        3D tensor with shape: `(batch_size, seq_len, dim)` where seq_len <= max_seq_len.

    Output shape:
        3D tensor with shape: `(batch_size, seq_len, dim)` (same as input).

    Raises:
        ValueError: If max_seq_len, dim, or scale are not positive.
        ValueError: If dropout is not in [0, 1].
        ValueError: If input dimension doesn't match expected dim during build.

    Example:
        ```python
        # Create positional embedding layer
        pos_embed = PositionalEmbedding(max_seq_len=512, dim=256)

        # Input sequence
        inputs = keras.Input(shape=(128, 256))  # seq_len=128, dim=256

        # Add positional information
        output = pos_embed(inputs)
        print(output.shape)  # (None, 128, 256)

        # With dropout for regularization
        pos_embed_dropout = PositionalEmbedding(
            max_seq_len=512,
            dim=256,
            dropout=0.1
        )
        output_dropout = pos_embed_dropout(inputs)

        # In a model
        model = keras.Model(inputs, output_dropout)
        ```
    """

    def __init__(
            self,
            max_seq_len: int,
            dim: int,
            dropout: float = 0.0,
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
        if not (0.0 <= dropout <= 1.0):
            raise ValueError(f"dropout must be in [0, 1], got {dropout}")
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")

        # Store ALL configuration parameters
        self.max_seq_len = max_seq_len
        self.dim = dim
        self.dropout_rate = dropout
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
        self.dropout = layers.Dropout(self.dropout_rate, name="pos_dropout")

        # Weight will be initialized in build()
        self.pos_embedding = None

        logger.info(f"Initialized PositionalEmbedding with max_seq_len={self.max_seq_len}, "
                    f"dim={self.dim}, dropout={self.dropout_rate}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer by creating weights and building sub-layers.

        Args:
            input_shape: Shape tuple of the input tensor (batch_size, seq_len, dim).
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

        Args:
            inputs: Input tensor with shape (batch_size, seq_len, dim).
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            Output tensor with positional embeddings added.
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

        Args:
            input_shape: Shape tuple of the input.

        Returns:
            Output shape tuple (same as input shape).
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        Returns:
            Dictionary containing ALL __init__ parameters for proper serialization.
        """
        config = super().get_config()
        config.update({
            "max_seq_len": self.max_seq_len,
            "dim": self.dim,
            "dropout": self.dropout_rate,
            "pos_initializer": keras.initializers.serialize(self.pos_initializer),
            "scale": self.scale,
        })
        return config


# ---------------------------------------------------------------------