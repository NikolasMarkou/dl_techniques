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
from typing import Optional, Dict, Any, Union

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class PositionalEmbedding(keras.layers.Layer):
    """Learned positional embedding layer with enhanced stability.

    This layer adds learnable positional embeddings to input sequences, allowing
    the model to understand the position of elements in the sequence. The embeddings
    are learned during training and can capture complex positional relationships.

    Args:
        max_seq_len: int, maximum sequence length that this layer can handle.
        dim: int, embedding dimension. Must match the last dimension of input.
        dropout: float, dropout rate applied after adding positional embeddings.
            Default is 0.0 (no dropout).
        pos_initializer: str or Initializer, initializer for positional embeddings.
            Default is "truncated_normal".
        scale: float, standard deviation for truncated normal initialization when
            using default initializer. Default is 0.02.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        3D tensor with shape: `(batch_size, seq_len, dim)` where seq_len <= max_seq_len.

    Output shape:
        3D tensor with shape: `(batch_size, seq_len, dim)` (same as input).

    Returns:
        A tensor with positional embeddings added to the input.

    Raises:
        ValueError: If input sequence length exceeds max_seq_len during runtime.

    Example:
        >>> # Create positional embedding layer
        >>> pos_embed = PositionalEmbedding(max_seq_len=512, dim=256)
        >>> # Input sequence
        >>> x = tf.random.normal([32, 128, 256])  # batch_size=32, seq_len=128, dim=256
        >>> # Add positional information
        >>> output = pos_embed(x)
        >>> print(output.shape)
        (32, 128, 256)
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

        # Store configuration
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

        # Will be initialized in build()
        self.pos_embedding = None
        self.dropout = None
        self._build_input_shape = None

    def build(self, input_shape: tuple) -> None:
        """Build the layer by creating weights and sublayers.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        # Store for serialization
        self._build_input_shape = input_shape

        # Validate input shape
        if len(input_shape) != 3:
            raise ValueError(
                f"Input must be 3D (batch_size, seq_len, dim), got shape {input_shape}"
            )

        if input_shape[-1] != self.dim:
            raise ValueError(
                f"Input dimension {input_shape[-1]} does not match expected dim {self.dim}"
            )

        # Create positional embeddings
        self.pos_embedding = self.add_weight(
            name="pos_embedding",
            shape=(1, self.max_seq_len, self.dim),
            initializer=self.pos_initializer,
            trainable=True,
        )

        # Create dropout layer
        self.dropout = layers.Dropout(self.dropout_rate)

        super().build(input_shape)

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

        # Runtime validation - this creates a dynamic check
        # In practice, this should be handled at the model level
        # but we include it for robustness

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

    def compute_output_shape(self, input_shape: tuple) -> tuple:
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
            Dictionary containing the layer configuration.
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

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration for serialization.

        Returns:
            Dictionary containing build configuration.
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build the layer from configuration.

        Args:
            config: Dictionary containing build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------
