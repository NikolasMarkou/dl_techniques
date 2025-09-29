"""
Qwen3 text embedding and reranking.

This module provides Keras 3 implementations of the Qwen3 text embedding and
reranking models, as detailed in the technical report "Qwen3 Embedding:
Advancing Text Embedding and Reranking Through Foundation Models". The
implementations are built using a factory-based, modular architecture,
allowing for configurable components like attention and normalization layers
while adhering to modern Keras patterns for robust serialization.

**1. Qwen3 Text Embedding: Instruction-Aware Semantic Vectors**

The embedding model leverages a causal transformer (decoder-only) architecture
to generate dense vector representations of text. Its design is distinguished
by several key principles that enhance performance on retrieval tasks:

-   **Instruction-Awareness**: The model is trained to be sensitive to task-
    specific instructions. For query-like inputs, a formatted instruction
    (e.g., "Instruct: Retrieve relevant passages...") is prepended to the
    text. This allows the model to adapt its embedding generation process to
    the specific semantic context of the task, such as retrieval, classification,
    or clustering. Documents are embedded without instructions to maintain a
    general representation.

-   **Last-Token Pooling**: Unlike BERT-style models that often use a special
    [CLS] token, the final embedding is derived from the hidden state of the
    last non-padding token in the sequence. In a causal model, this final state
    is theoretically positioned to aggregate information from the entire
    preceding sequence, making it a comprehensive representation.

-   **Matryoshka Representation Learning (MRL)**: The model supports generating
    embeddings of variable dimensions without retraining. By simply truncating
    the full-dimension embedding vector, users can obtain smaller, more
    efficient vectors that retain a high degree of semantic quality. This is
    achieved through specific training techniques that make the initial
    dimensions of the vector the most informative.

-   **L2 Normalization**: The final pooled and truncated vector is L2-normalized
    to produce a unit vector. This standardizes the embeddings, making them
    directly suitable for efficient similarity calculations using cosine
    similarity or maximum inner product search (MIPS).

**2. Qwen3 Reranking: Generative Relevance Classification**

The reranker model reframes the task of relevance scoring from a simple
similarity calculation to a generative classification problem. It uses the
advanced reasoning and context-understanding capabilities of a causal
language model to make a nuanced judgment about a query-document pair.

-   **Prompt-Based Judgment**: Instead of comparing two separate embeddings, the
    reranker processes a single, carefully crafted prompt that includes the
    instruction, the query, and the document. The prompt concludes by framing
    the task as a question for the model to answer, asking it to generate either
    "yes" or "no" to indicate relevance.

-   **Probabilistic Scoring**: The final relevance score is not a simple logit
    but the model's confidence in its judgment. It is calculated as the
    softmax probability of the "yes" token over the logits of the "yes" and
    "no" tokens. This approach directly leverages the model's generative
    pre-training to produce a more reliable and context-aware relevance score.
    Mathematically, this is expressed as:
        Score = P("yes" | prompt) = softmax(logits["yes"], logits["no"])[1]

**Foundational References:**

-   Zhang, Y., et al. (2025). *Qwen3 Embedding: Advancing Text Embedding and
    Reranking Through Foundation Models*. arXiv:2506.05176.
-   Vaswani, A., et al. (2017). *Attention Is All You Need*. arXiv:1706.03762.
"""

import keras
from keras import ops, layers
from typing import Optional, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.transformer import TransformerLayer
from dl_techniques.layers.embedding.factory import create_embedding_layer
from dl_techniques.layers.norms.factory import create_normalization_layer

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class Qwen3EmbeddingLayer(keras.layers.Layer):
    """
    Keras implementation of the Qwen3 Text Embedding model using factory components.

    This layer implements a modern transformer-based text embedding architecture
    using configurable components from the dl_techniques framework. It processes
    tokenized text through multiple transformer layers and applies last-token pooling
    with optional L2 normalization and Matryoshka Representation Learning (MRL).

    **Intent**: Provide a reusable, configurable Keras Layer for text embedding
    that leverages modern architectural components while maintaining full
    serialization compatibility.

    **Architecture**:
    ```
    Input Tokens -> Token Embeddings -> Positional Embeddings -> 
    N × TransformerLayer -> Last Token Pooling -> Optional Truncation -> 
    L2 Normalization -> Output Embedding
    ```

    **Mathematical Operation**:
        embedding = normalize(hidden_state[:, last_token_idx, :truncate_dim])

    Where `last_token_idx` is determined from the attention mask.

    Args:
        vocab_size (int): Size of the vocabulary for token embeddings.
        hidden_size (int): Dimension of the hidden representations throughout the model.
        num_layers (int): Number of transformer layers to stack.
        num_heads (int): Number of attention heads in each transformer layer.
        intermediate_size (int): Size of the intermediate layer in FFN blocks.
        max_seq_len (int): Maximum sequence length for positional embeddings.
        normalize (bool): If True, applies L2 normalization to final embeddings.
        truncate_dim (Optional[int]): If set, truncates embeddings to this dimension
            for Matryoshka Representation Learning (MRL).
        dropout_rate (float): Dropout rate applied throughout the model.
        ffn_type (str): Type of FFN to use ('mlp', 'swiglu', 'geglu', etc.).
        normalization_type (str): Type of normalization ('layer_norm', 'rms_norm', etc.).
        attention_type (str): Type of attention mechanism to use.
        **kwargs: Additional arguments for the base Layer class.

    Input shape:
        A dictionary containing:
        - 'input_ids': Tensor of shape `(batch_size, sequence_length)`.
        - 'attention_mask': Tensor of shape `(batch_size, sequence_length)`.

    Output shape:
        A 2D tensor of shape `(batch_size, embedding_dimension)`, where
        `embedding_dimension` is `hidden_size` or `truncate_dim` if specified.

    Example:
        ```python
        # Create embedding layer
        embedding_layer = Qwen3EmbeddingLayer(
            vocab_size=32000,
            hidden_size=1024,
            num_layers=12,
            num_heads=16,
            intermediate_size=2816,
            max_seq_len=8192,
            truncate_dim=256
        )

        # Process tokenized inputs
        inputs = {
            'input_ids': tf.constant([[1, 2, 3, 4, 0]]),  # With padding
            'attention_mask': tf.constant([[1, 1, 1, 1, 0]])
        }
        embeddings = embedding_layer(inputs)  # Shape: (1, 256)
        ```

    Note:
        This layer builds its sub-layers in the `build()` method following
        modern Keras 3 patterns for proper serialization support.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 1024,
        num_layers: int = 12,
        num_heads: int = 16,
        intermediate_size: int = 2816,
        max_seq_len: int = 8192,
        normalize: bool = True,
        truncate_dim: Optional[int] = None,
        dropout_rate: float = 0.0,
        ffn_type: str = 'swiglu',
        normalization_type: str = 'rms_norm',
        attention_type: str = 'multi_head',
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Store configuration
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.max_seq_len = max_seq_len
        self.normalize = normalize
        self.truncate_dim = truncate_dim
        self.dropout_rate = dropout_rate
        self.ffn_type = ffn_type
        self.normalization_type = normalization_type
        self.attention_type = attention_type

        # Create sub-layers in __init__
        self.token_embeddings = layers.Embedding(
            vocab_size,
            hidden_size,
            name='token_embeddings'
        )

        self.positional_embeddings = create_embedding_layer(
            'positional_learned',
            max_seq_len=max_seq_len,
            dim=hidden_size,
            name='positional_embeddings'
        )

        self.embedding_dropout = layers.Dropout(dropout_rate, name='embedding_dropout')

        # Create transformer layers
        self.transformer_layers = []
        for i in range(num_layers):
            transformer_layer = TransformerLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                attention_type=attention_type,
                ffn_type=ffn_type,
                normalization_type=normalization_type,
                dropout_rate=dropout_rate,
                name=f'transformer_layer_{i}'
            )
            self.transformer_layers.append(transformer_layer)

        # Final layer norm
        self.final_norm = create_normalization_layer(
            normalization_type,
            name='final_norm'
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build all sub-layers with proper shapes."""
        # Get batch size and sequence length from input shape
        # input_shape represents the shape of input_ids: (batch_size, seq_len)
        batch_size = input_shape.get('input_ids', [None, None])[0] if isinstance(input_shape, dict) else None
        seq_len = input_shape.get('input_ids', [None, None])[1] if isinstance(input_shape, dict) else None

        # Build token embeddings
        self.token_embeddings.build((batch_size, seq_len))

        # Build positional embeddings - expects (batch_size, seq_len, hidden_size)
        self.positional_embeddings.build((batch_size, seq_len, self.hidden_size))

        # Build dropout layer
        self.embedding_dropout.build((batch_size, seq_len, self.hidden_size))

        # Build transformer layers
        transformer_input_shape = (batch_size, seq_len, self.hidden_size)
        for transformer_layer in self.transformer_layers:
            transformer_layer.build(transformer_input_shape)

        # Build final normalization
        self.final_norm.build(transformer_input_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: Dict[str, keras.KerasTensor],
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass to compute embeddings."""
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # Token embeddings
        hidden_states = self.token_embeddings(input_ids)

        # Add positional embeddings
        hidden_states = self.positional_embeddings(hidden_states)

        # Apply dropout
        hidden_states = self.embedding_dropout(hidden_states, training=training)

        # Process through transformer layers
        for transformer_layer in self.transformer_layers:
            hidden_states = transformer_layer(
                hidden_states,
                attention_mask=attention_mask,
                training=training
            )

        # Apply final normalization
        hidden_states = self.final_norm(hidden_states)

        # Last-token pooling
        sequence_lengths = ops.sum(ops.cast(attention_mask, "int32"), axis=1) - 1
        batch_size = ops.shape(hidden_states)[0]

        # Create indices for gathering the last token's hidden state
        batch_indices = ops.arange(batch_size)
        gather_indices = ops.stack([batch_indices, sequence_lengths], axis=1)

        pooled_embeddings = ops.take_along_axis(
            hidden_states,
            ops.expand_dims(gather_indices[:, 1], axis=-1),
            axis=1
        )
        pooled_embeddings = ops.squeeze(pooled_embeddings, axis=1)

        # Apply optional dimension truncation (MRL)
        if self.truncate_dim:
            pooled_embeddings = pooled_embeddings[:, :self.truncate_dim]

        # Apply optional L2 normalization
        if self.normalize:
            pooled_embeddings = ops.normalize(pooled_embeddings, axis=1)

        return pooled_embeddings

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'intermediate_size': self.intermediate_size,
            'max_seq_len': self.max_seq_len,
            'normalize': self.normalize,
            'truncate_dim': self.truncate_dim,
            'dropout_rate': self.dropout_rate,
            'ffn_type': self.ffn_type,
            'normalization_type': self.normalization_type,
            'attention_type': self.attention_type,
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class Qwen3RerankerLayer(keras.layers.Layer):
    """
    Keras implementation of the Qwen3 Reranker using factory components.

    This layer implements a causal language model for text reranking by computing
    the probability of generating "yes" tokens given query-document pairs formatted
    as special prompts. It uses configurable transformer architecture components.

    **Intent**: Provide a core, serializable Keras Layer for text reranking
    that can be integrated into larger ranking and retrieval systems.

    **Architecture**:
    ```
    Formatted Prompt -> Token Embeddings -> Positional Embeddings ->
    N × TransformerLayer -> Language Modeling Head -> Logits["no", "yes"] ->
    Softmax -> Score
    ```

    **Mathematical Operation**:
        score = Softmax(logits[last_token_idx, [no_id, yes_id]])[1]

    Args:
        vocab_size (int): Size of the vocabulary.
        hidden_size (int): Hidden dimension throughout the model.
        num_layers (int): Number of transformer layers.
        num_heads (int): Number of attention heads.
        intermediate_size (int): Size of the FFN intermediate layer.
        max_seq_len (int): Maximum sequence length.
        dropout_rate (float): Dropout rate applied throughout the model.
        ffn_type (str): Type of FFN to use.
        normalization_type (str): Type of normalization to use.
        attention_type (str): Type of attention mechanism.
        yes_token_id (int): Token ID for "yes" in the vocabulary.
        no_token_id (int): Token ID for "no" in the vocabulary.
        **kwargs: Additional arguments for the base Layer class.

    Input shape:
        A dictionary containing:
        - 'input_ids': Tensor of shape `(batch_size, sequence_length)`.
        - 'attention_mask': Tensor of shape `(batch_size, sequence_length)`.

    Output shape:
        A 1D tensor of shape `(batch_size,)` containing relevance scores
        between 0 and 1.

    Example:
        ```python
        # Create reranker layer
        reranker_layer = Qwen3RerankerLayer(
            vocab_size=32000,
            hidden_size=1024,
            num_layers=12,
            num_heads=16,
            yes_token_id=9891,  # "yes" token ID
            no_token_id=2201    # "no" token ID
        )

        # Process formatted prompts
        inputs = {
            'input_ids': tf.constant([[1, 2, 3, 4, 5]]),
            'attention_mask': tf.constant([[1, 1, 1, 1, 1]])
        }
        scores = reranker_layer(inputs)  # Shape: (1,)
        ```
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 1024,
        num_layers: int = 12,
        num_heads: int = 16,
        intermediate_size: int = 2816,
        max_seq_len: int = 8192,
        dropout_rate: float = 0.0,
        ffn_type: str = 'swiglu',
        normalization_type: str = 'rms_norm',
        attention_type: str = 'multi_head',
        yes_token_id: int = 9891,
        no_token_id: int = 2201,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Store configuration
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout_rate
        self.ffn_type = ffn_type
        self.normalization_type = normalization_type
        self.attention_type = attention_type
        self.yes_token_id = yes_token_id
        self.no_token_id = no_token_id

        # Create sub-layers in __init__
        self.token_embeddings = layers.Embedding(
            vocab_size,
            hidden_size,
            name='token_embeddings'
        )

        self.positional_embeddings = create_embedding_layer(
            'positional_learned',
            max_seq_len=max_seq_len,
            dim=hidden_size,
            name='positional_embeddings'
        )

        self.embedding_dropout = layers.Dropout(dropout_rate, name='embedding_dropout')

        # Create transformer layers
        self.transformer_layers = []
        for i in range(num_layers):
            transformer_layer = TransformerLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                attention_type=attention_type,
                ffn_type=ffn_type,
                normalization_type=normalization_type,
                dropout_rate=dropout_rate,
                name=f'transformer_layer_{i}'
            )
            self.transformer_layers.append(transformer_layer)

        # Final layer norm
        self.final_norm = create_normalization_layer(
            normalization_type,
            name='final_norm'
        )

        # Language modeling head
        self.lm_head = layers.Dense(
            vocab_size,
            use_bias=False,
            name='lm_head'
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build all sub-layers with proper shapes."""
        # Get shapes from input
        batch_size = input_shape.get('input_ids', [None, None])[0] if isinstance(input_shape, dict) else None
        seq_len = input_shape.get('input_ids', [None, None])[1] if isinstance(input_shape, dict) else None

        # Build token embeddings
        self.token_embeddings.build((batch_size, seq_len))

        # Build positional embeddings
        self.positional_embeddings.build((batch_size, seq_len, self.hidden_size))

        # Build dropout layer
        self.embedding_dropout.build((batch_size, seq_len, self.hidden_size))

        # Build transformer layers
        transformer_input_shape = (batch_size, seq_len, self.hidden_size)
        for transformer_layer in self.transformer_layers:
            transformer_layer.build(transformer_input_shape)

        # Build final normalization
        self.final_norm.build(transformer_input_shape)

        # Build language modeling head
        self.lm_head.build((batch_size, seq_len, self.hidden_size))

        super().build(input_shape)

    def call(
        self,
        inputs: Dict[str, keras.KerasTensor],
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass to compute relevance scores."""
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # Token embeddings
        hidden_states = self.token_embeddings(input_ids)

        # Add positional embeddings
        hidden_states = self.positional_embeddings(hidden_states)

        # Apply dropout
        hidden_states = self.embedding_dropout(hidden_states, training=training)

        # Process through transformer layers
        for transformer_layer in self.transformer_layers:
            hidden_states = transformer_layer(
                hidden_states,
                attention_mask=attention_mask,
                training=training
            )

        # Apply final normalization
        hidden_states = self.final_norm(hidden_states)

        # Get logits from language modeling head
        logits = self.lm_head(hidden_states)

        # Get the logits for the last token in each sequence
        sequence_lengths = ops.sum(ops.cast(attention_mask, "int32"), axis=1) - 1
        batch_size = ops.shape(logits)[0]

        # Create indices for gathering
        batch_indices = ops.arange(batch_size)
        gather_indices = ops.stack([batch_indices, sequence_lengths], axis=1)

        # Gather last token logits
        last_token_logits = ops.take_along_axis(
            logits,
            ops.expand_dims(gather_indices[:, 1], axis=-1),
            axis=1
        )
        last_token_logits = ops.squeeze(last_token_logits, axis=1)

        # Extract logits for "yes" and "no" tokens
        yes_logits = last_token_logits[:, self.yes_token_id]
        no_logits = last_token_logits[:, self.no_token_id]

        # Compute the score via softmax
        combined_logits = ops.stack([no_logits, yes_logits], axis=1)
        probabilities = keras.activations.softmax(combined_logits, axis=1)

        # The score is the probability of "yes"
        scores = probabilities[:, 1]

        return scores

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'intermediate_size': self.intermediate_size,
            'max_seq_len': self.max_seq_len,
            'dropout_rate': self.dropout_rate,
            'ffn_type': self.ffn_type,
            'normalization_type': self.normalization_type,
            'attention_type': self.attention_type,
            'yes_token_id': self.yes_token_id,
            'no_token_id': self.no_token_id,
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class Qwen3EmbeddingModel(keras.Model):
    """
    High-level Keras Model for Qwen3 Text Embedding.

    This model provides a user-friendly interface for generating text embeddings
    with support for different instruction types and document processing modes.
    It wraps the `Qwen3EmbeddingLayer` and provides convenient methods for
    processing queries and documents.

    **Intent**: To offer a simple, `compile()`- and `fit()`-ready Keras Model
    that abstracts tokenization details while providing flexible embedding
    generation for various text types.

    Args:
        vocab_size (int): Size of the vocabulary for token embeddings.
        hidden_size (int): Dimension of hidden representations.
        num_layers (int): Number of transformer layers.
        num_heads (int): Number of attention heads.
        intermediate_size (int): Size of FFN intermediate layer.
        max_seq_len (int): Maximum sequence length.
        normalize (bool): Whether to L2-normalize embeddings.
        truncate_dim (Optional[int]): Optional dimension for MRL.
        dropout_rate (float): Dropout rate throughout the model.
        ffn_type (str): Type of FFN to use.
        normalization_type (str): Type of normalization.
        attention_type (str): Type of attention mechanism.
        **kwargs: Additional arguments for the base Model class.

    Example:
        ```python
        model = Qwen3EmbeddingModel(
            vocab_size=32000,
            hidden_size=1024,
            num_layers=12,
            num_heads=16,
            truncate_dim=256
        )

        # Create sample tokenized inputs
        inputs = {
            'input_ids': tf.constant([[1, 2, 3, 4, 0]]),
            'attention_mask': tf.constant([[1, 1, 1, 1, 0]])
        }

        embeddings = model(inputs)  # Shape: (1, 256)
        ```
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 1024,
        num_layers: int = 12,
        num_heads: int = 16,
        intermediate_size: int = 2816,
        max_seq_len: int = 8192,
        normalize: bool = True,
        truncate_dim: Optional[int] = None,
        dropout_rate: float = 0.0,
        ffn_type: str = 'swiglu',
        normalization_type: str = 'rms_norm',
        attention_type: str = 'multi_head',
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Store configuration
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.max_seq_len = max_seq_len
        self.normalize = normalize
        self.truncate_dim = truncate_dim
        self.dropout_rate = dropout_rate
        self.ffn_type = ffn_type
        self.normalization_type = normalization_type
        self.attention_type = attention_type

        # Create the underlying embedding layer
        self.embedding_layer = Qwen3EmbeddingLayer(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            max_seq_len=max_seq_len,
            normalize=normalize,
            truncate_dim=truncate_dim,
            dropout_rate=dropout_rate,
            ffn_type=ffn_type,
            normalization_type=normalization_type,
            attention_type=attention_type,
            name="qwen3_embedding_layer"
        )

    def call(
        self,
        inputs: Dict[str, keras.KerasTensor],
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """The base forward pass expects tokenized inputs."""
        return self.embedding_layer(inputs, training=training)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'intermediate_size': self.intermediate_size,
            'max_seq_len': self.max_seq_len,
            'normalize': self.normalize,
            'truncate_dim': self.truncate_dim,
            'dropout_rate': self.dropout_rate,
            'ffn_type': self.ffn_type,
            'normalization_type': self.normalization_type,
            'attention_type': self.attention_type,
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class Qwen3RerankerModel(keras.Model):
    """
    High-level Keras Model for Qwen3 Text Reranking.

    This model provides a user-friendly interface for computing relevance scores
    between query-document pairs. It wraps the `Qwen3RerankerLayer` and provides
    methods for processing formatted reranking prompts.

    **Intent**: To offer a simple, end-to-end interface for reranking tasks
    that can be easily integrated into retrieval and ranking pipelines.

    Args:
        vocab_size (int): Size of the vocabulary.
        hidden_size (int): Hidden dimension throughout the model.
        num_layers (int): Number of transformer layers.
        num_heads (int): Number of attention heads.
        intermediate_size (int): Size of FFN intermediate layer.
        max_seq_len (int): Maximum sequence length.
        dropout_rate (float): Dropout rate throughout the model.
        ffn_type (str): Type of FFN to use.
        normalization_type (str): Type of normalization.
        attention_type (str): Type of attention mechanism.
        yes_token_id (int): Token ID for "yes".
        no_token_id (int): Token ID for "no".
        **kwargs: Additional arguments for the base Model class.

    Example:
        ```python
        reranker = Qwen3RerankerModel(
            vocab_size=32000,
            hidden_size=1024,
            num_layers=12,
            num_heads=16,
            yes_token_id=9891,
            no_token_id=2201
        )

        # Create sample formatted inputs
        inputs = {
            'input_ids': tf.constant([[1, 2, 3, 4, 5]]),
            'attention_mask': tf.constant([[1, 1, 1, 1, 1]])
        }

        scores = reranker(inputs)  # Shape: (1,)
        ```
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 1024,
        num_layers: int = 12,
        num_heads: int = 16,
        intermediate_size: int = 2816,
        max_seq_len: int = 8192,
        dropout_rate: float = 0.0,
        ffn_type: str = 'swiglu',
        normalization_type: str = 'rms_norm',
        attention_type: str = 'multi_head',
        yes_token_id: int = 9891,
        no_token_id: int = 2201,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Store configuration
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout_rate
        self.ffn_type = ffn_type
        self.normalization_type = normalization_type
        self.attention_type = attention_type
        self.yes_token_id = yes_token_id
        self.no_token_id = no_token_id

        # Create the underlying reranker layer
        self.reranker_layer = Qwen3RerankerLayer(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            max_seq_len=max_seq_len,
            dropout_rate=dropout_rate,
            ffn_type=ffn_type,
            normalization_type=normalization_type,
            attention_type=attention_type,
            yes_token_id=yes_token_id,
            no_token_id=no_token_id,
            name="qwen3_reranker_layer"
        )

    def call(
        self,
        inputs: Dict[str, keras.KerasTensor],
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """The base forward pass expects tokenized, pre-formatted inputs."""
        return self.reranker_layer(inputs, training=training)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'intermediate_size': self.intermediate_size,
            'max_seq_len': self.max_seq_len,
            'dropout_rate': self.dropout_rate,
            'ffn_type': self.ffn_type,
            'normalization_type': self.normalization_type,
            'attention_type': self.attention_type,
            'yes_token_id': self.yes_token_id,
            'no_token_id': self.no_token_id,
        })
        return config

# ---------------------------------------------------------------------
