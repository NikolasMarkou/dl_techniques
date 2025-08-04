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
from typing import Optional, Union

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class SparsePuzzleEmbedding(keras.layers.Layer):
    """
    Sparse embedding layer for puzzle identifiers.

    This layer creates embeddings for puzzle identifiers that are updated
    sparsely during training. During inference, it uses the full embedding table.

    Args:
        num_embeddings: Number of puzzle identifiers
        embedding_dim: Dimension of embeddings
        batch_size: Batch size for training (used for local embeddings)
        embeddings_initializer: Initializer for embedding weights
        embeddings_regularizer: Regularizer for embedding weights
        **kwargs: Additional layer arguments
    """

    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            batch_size: int,
            embeddings_initializer: Union[str, keras.initializers.Initializer] = "zeros",
            embeddings_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.embeddings_initializer = embeddings_initializer
        self.embeddings_regularizer = embeddings_regularizer

        # Main embedding table
        self.embeddings = self.add_weight(
            name="embeddings",
            shape=(num_embeddings, embedding_dim),
            initializer=self.embeddings_initializer,
            regularizer=self.embeddings_regularizer,
            trainable=True
        )

        # Local embeddings for training (non-trainable, used for caching)
        self.local_embeddings = self.add_weight(
            name="local_embeddings",
            shape=(batch_size, embedding_dim),
            initializer="zeros",
            trainable=False
        )

        # Local IDs for tracking which embeddings are cached
        self.local_ids = self.add_weight(
            name="local_ids",
            shape=(batch_size,),
            initializer="zeros",
            trainable=False,
            dtype="int32"
        )

    def build(self, input_shape):
        """Build layer."""
        super().build(input_shape)

    def call(self, inputs, training=None):
        """
        Forward pass through sparse embedding.

        Args:
            inputs: Puzzle identifier indices (batch_size,)
            training: Whether in training mode

        Returns:
            Embeddings for the puzzle identifiers (batch_size, embedding_dim)
        """
        if not training:
            # During inference, directly use the embedding table
            return keras.ops.take(self.embeddings, inputs, axis=0)

        # During training, cache embeddings in local storage for efficiency
        # This mimics the sparse embedding behavior from the original code

        # Copy embeddings from main table to local cache
        embeddings = keras.ops.take(self.embeddings, inputs, axis=0)

        # Update local cache (for potential sparse updates later)
        self.local_embeddings.assign(embeddings)
        self.local_ids.assign(keras.ops.cast(inputs, "int32"))

        return embeddings

    def compute_output_shape(self, input_shape):
        """Compute output shape."""
        return input_shape + (self.embedding_dim,)

    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "num_embeddings": self.num_embeddings,
            "embedding_dim": self.embedding_dim,
            "batch_size": self.batch_size,
            "embeddings_initializer": keras.initializers.serialize(self.embeddings_initializer),
            "embeddings_regularizer": keras.regularizers.serialize(self.embeddings_regularizer),
        })
        return config

    def get_build_config(self):
        """Get build configuration."""
        return {"input_shape": getattr(self, "_build_input_shape", None)}

    def build_from_config(self, config):
        """Build from configuration."""
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------

