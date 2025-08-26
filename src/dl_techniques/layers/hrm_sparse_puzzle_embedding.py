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
    Sparse embedding layer optimized for large-scale puzzle identifier lookups with training efficiency.

    This layer implements a specialized embedding mechanism designed for scenarios with very large
    vocabulary sizes (e.g., millions of puzzle identifiers) where sparse updates are crucial for
    training efficiency. It maintains a dual-mode operation that optimizes both inference speed
    and training memory usage through intelligent caching mechanisms.

    **Intent**: Provide memory-efficient embedding lookups for large categorical vocabularies,
    specifically designed for puzzle-based reasoning tasks where only a small subset of embeddings
    are actively updated in each training batch, enabling scalable training on massive datasets.

    **Architecture & Operation**:
    ```
    Training Mode (training=True):
    Input IDs → Embedding Lookup → Cache (local_embeddings, local_ids) → Output Embeddings

    Inference Mode (training=False):
    Input IDs → Direct Embedding Lookup → Output Embeddings
    ```

    **Dual-Mode Behavior**:

    **Training Mode**:
    - Performs standard embedding lookup from main table
    - **Caches** current batch embeddings and IDs in local non-trainable variables
    - Enables external sparse gradient update mechanisms
    - Optimizes memory usage by isolating active embeddings

    **Inference Mode**:
    - Direct lookup from main embedding table
    - No caching overhead for maximum speed
    - Standard embedding layer behavior

    **Training Optimization**:
    The caching mechanism enables external training loops to:
    - Gather gradients only for active embeddings (local_embeddings)
    - Apply sparse updates directly to main table using cached IDs
    - Avoid full embedding matrix operations during backpropagation

    Args:
        num_embeddings: Integer, total number of puzzle identifiers in vocabulary.
            Must be positive. This defines the size of the main embedding table.
        embedding_dim: Integer, dimensionality of embedding vectors.
            Must be positive. All embeddings will have this dimension.
        batch_size: Integer, expected batch size for training.
            Must be positive. Used to size local caching arrays for efficiency.
        embeddings_initializer: String or Initializer, method to initialize main embeddings.
            Defaults to 'zeros'. Consider 'random_normal' for better initial diversity.
        embeddings_regularizer: Optional Regularizer, L1/L2 regularization for embeddings.
            Helps prevent overfitting on large embedding tables. Defaults to None.
        **kwargs: Additional Layer base class arguments.

    Input shape:
        1D integer tensor with shape: `(batch_size,)`.
        Values should be valid indices in range [0, num_embeddings).

    Output shape:
        2D tensor with shape: `(batch_size, embedding_dim)`.
        Each input ID mapped to its corresponding embedding vector.

    Attributes:
        embeddings: Main trainable embedding table, shape (num_embeddings, embedding_dim).
        local_embeddings: Non-trainable cache for current batch embeddings during training.
        local_ids: Non-trainable cache for current batch IDs during training.

    Example:
        ```python
        # Large-scale puzzle embedding (1M puzzles, 768-dim embeddings)
        embedding_layer = SparsePuzzleEmbedding(
            num_embeddings=1_000_000,
            embedding_dim=768,
            batch_size=32,
            embeddings_initializer='random_normal'
        )

        # Puzzle IDs as input
        puzzle_ids = keras.random.randint((32,), 0, 1_000_000)  # Random puzzle IDs
        embeddings = embedding_layer(puzzle_ids, training=True)
        print(embeddings.shape)  # (32, 768)

        # Inference mode (no caching overhead)
        test_embeddings = embedding_layer(puzzle_ids, training=False)

        # Access cached data for sparse updates (during training)
        if hasattr(embedding_layer, 'local_embeddings'):
            cached_embeddings = embedding_layer.local_embeddings  # Current batch embeddings
            cached_ids = embedding_layer.local_ids              # Corresponding IDs
        ```

    Note:
        This layer is optimized for training scenarios where external mechanisms handle
        sparse gradient updates. The caching functionality provides the necessary data
        structures for implementing efficient sparse training loops while maintaining
        standard Keras layer compatibility.
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

        This method creates:
        1. Main trainable embedding table for all puzzle identifiers
        2. Local non-trainable cache for current batch embeddings (training optimization)
        3. Local non-trainable cache for current batch IDs (training optimization)

        Args:
            input_shape: Shape of input tensor, expected to be (batch_size,).
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

        Args:
            inputs: Integer tensor of puzzle IDs with shape (batch_size,).
                    Values must be in range [0, num_embeddings).
            training: Boolean indicating training mode. When True, enables caching
                     for sparse update optimization.

        Returns:
            Embedding tensor with shape (batch_size, embedding_dim).
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

        Args:
            input_shape: Input shape tuple (batch_size,).

        Returns:
            Output shape tuple (batch_size, embedding_dim).
        """
        return input_shape + (self.embedding_dim,)

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration for serialization.

        Returns:
            Dictionary containing all initialization parameters.
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