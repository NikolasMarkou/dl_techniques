"""
This module defines the SparsePuzzleEmbedding layer.

This layer provides vector representations (embeddings) for a large set of categorical
puzzle identifiers. Its key feature is its specialized handling of embeddings during
training to facilitate efficient, sparse updates, which is particularly useful when
the total number of embeddings (`num_embeddings`) is very large.

The layer operates in two distinct modes:
1.  **Inference Mode (`training=False`):**
    It functions as a standard embedding lookup table. Given a batch of puzzle IDs,
    it directly retrieves the corresponding embedding vectors from the main, fully
    trained embedding matrix.

2.  **Training Mode (`training=True`):**
    The layer is designed to work with a training process that performs sparse
    gradient updates (e.g., in a custom `train_step`). In its forward pass, it
    performs a standard lookup but also explicitly caches the embeddings and their
    corresponding IDs for the current batch into non-trainable local variables
    (`local_embeddings` and `local_ids`).

The purpose of this caching mechanism is to isolate the batch-specific embeddings
that are actively being trained. This allows an external training mechanism to
potentially gather gradients and apply them in a more optimized scatter-update
operation directly to the main embedding table, avoiding operations on the
entire (and potentially massive) weight matrix. The layer itself prepares the
necessary components for this optimization but does not implement the sparse
update logic in its `call` method.

Internal State:
-   `embeddings`: The primary trainable weight matrix of shape (`num_embeddings`, `embedding_dim`)
    that stores all puzzle embeddings.
-   `local_embeddings`: A non-trainable cache to hold the embedding vectors for the
    puzzles in the current training batch.
-   `local_ids`: A non-trainable cache to hold the integer identifiers for the
    puzzles in the current training batch, corresponding to the `local_embeddings`.
"""

import keras
from typing import Optional, Union, Tuple, Any, Dict

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class SparsePuzzleEmbedding(keras.layers.Layer):
    """
    Sparse embedding layer optimized for large-scale puzzle identifier lookups.

    Implements a dual-mode embedding mechanism for very large vocabularies (e.g.,
    millions of puzzle identifiers). In training mode, it caches the current batch
    embeddings and IDs into local non-trainable variables (``local_embeddings``,
    ``local_ids``), enabling external sparse gradient updates that avoid operations
    on the full embedding matrix. In inference mode, it performs direct lookup
    without caching overhead.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────────────┐
        │         SparsePuzzleEmbedding                │
        │                                              │
        │  Training Mode:                              │
        │  Input IDs ──► take(embeddings, ids)         │
        │                    │                         │
        │                    ├──► Cache local_emb/ids  │
        │                    ▼                         │
        │              Output Embeddings               │
        │                                              │
        │  Inference Mode:                             │
        │  Input IDs ──► take(embeddings, ids)         │
        │                    │                         │
        │                    ▼                         │
        │              Output Embeddings               │
        └──────────────────────────────────────────────┘

    :param num_embeddings: Total number of puzzle identifiers in vocabulary.
    :type num_embeddings: int
    :param embedding_dim: Dimensionality of embedding vectors.
    :type embedding_dim: int
    :param batch_size: Expected batch size for training (sizes local caches).
    :type batch_size: int
    :param embeddings_initializer: Method to initialize main embeddings.
        Defaults to ``'zeros'``.
    :type embeddings_initializer: Union[str, keras.initializers.Initializer]
    :param embeddings_regularizer: Optional regularizer for embeddings.
        Defaults to None.
    :type embeddings_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional Layer base class arguments.
    :type kwargs: Any
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        batch_size: int,
        embeddings_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        embeddings_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if num_embeddings <= 0:
            raise ValueError(f"num_embeddings must be positive, got {num_embeddings}")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        # Store configuration
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.embeddings_initializer = keras.initializers.get(embeddings_initializer)
        self.embeddings_regularizer = keras.regularizers.get(embeddings_regularizer)

        # Initialize weight attributes - created in build()
        self.embeddings = None
        self.local_embeddings = None
        self.local_ids = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create the embedding weights and caching variables.

        :param input_shape: Shape of input tensor, expected ``(batch_size,)``.
        :type input_shape: Tuple[Optional[int], ...]
        """
        # Validate input shape
        if len(input_shape) != 1:
            raise ValueError(f"Expected 1D input (batch_size,), got shape {input_shape}")

        # Main embedding table - trainable weights for all puzzle identifiers
        self.embeddings = self.add_weight(
            name="embeddings",
            shape=(self.num_embeddings, self.embedding_dim),
            initializer=self.embeddings_initializer,
            regularizer=self.embeddings_regularizer,
            trainable=True
        )

        # Local embeddings cache for current batch during training - non-trainable
        self.local_embeddings = self.add_weight(
            name="local_embeddings",
            shape=(self.batch_size, self.embedding_dim),
            initializer="zeros",
            trainable=False
        )

        # Local IDs cache for tracking which embeddings are cached - non-trainable
        self.local_ids = self.add_weight(
            name="local_ids",
            shape=(self.batch_size,),
            initializer="zeros",
            trainable=False,
            dtype="int32"
        )

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through sparse embedding with mode-dependent behavior.

        :param inputs: Integer tensor of puzzle IDs with shape ``(batch_size,)``.
        :type inputs: keras.KerasTensor
        :param training: Whether in training mode (enables caching).
        :type training: Optional[bool]
        :return: Embedding tensor with shape ``(batch_size, embedding_dim)``.
        :rtype: keras.KerasTensor
        """
        # Validate input range (basic check)
        inputs = keras.ops.cast(inputs, "int32")

        if not training:
            # Inference Mode: Direct lookup from main embedding table
            # No caching overhead for maximum performance
            return keras.ops.take(self.embeddings, inputs, axis=0)

        # Training Mode: Lookup + caching for sparse update optimization

        # Standard embedding lookup from main table
        embeddings_output = keras.ops.take(self.embeddings, inputs, axis=0)

        # Cache current batch embeddings and IDs for external sparse updates
        # This enables external training mechanisms to:
        # 1. Gather gradients only for these specific embeddings
        # 2. Apply sparse updates directly to main table using cached IDs
        # 3. Avoid operations on the entire embedding matrix

        self.local_embeddings.assign(embeddings_output)
        self.local_ids.assign(inputs)

        return embeddings_output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute output shape by appending embedding dimension.

        :param input_shape: Input shape tuple ``(batch_size,)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple ``(batch_size, embedding_dim)``.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape + (self.embedding_dim,)

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration for serialization.

        :return: Dictionary containing all initialization parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "num_embeddings": self.num_embeddings,
            "embedding_dim": self.embedding_dim,
            "batch_size": self.batch_size,
            "embeddings_initializer": keras.initializers.serialize(self.embeddings_initializer),
            "embeddings_regularizer": keras.regularizers.serialize(self.embeddings_regularizer),
        })
        return config

# ---------------------------------------------------------------------