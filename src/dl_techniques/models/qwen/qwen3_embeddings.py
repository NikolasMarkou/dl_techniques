"""
Qwen3-style text embedding and reranking towers: a shared transformer trunk read out
either as a pooled vector or as a two-token relevance judgment.

Retrieval and reranking answer the same question at different budgets. An embedding
model must reduce a passage to one vector *before* it knows what will be asked of
it, so every document in a corpus can be encoded once and searched by inner product;
the compression is the whole point and also the whole limitation. A reranker is
allowed to read the query and the document together and spend a full forward pass
per pair, so it can model interactions that no pair of independent vectors can
represent. The two classes here share a trunk and differ only in the readout, which
makes the trade explicit: one reads a hidden state, the other reads the language
modelling head.

The embedding readout takes the hidden state at the last non-padding position rather
than a prepended `[CLS]`. In a decoder-only model that position is the only one that
has seen the entire sequence, which is the argument for the choice. The index is
computed as `sum(attention_mask) - 1`, and that arithmetic assumes the padding is on
the *right*: it counts real tokens rather than locating them, so a left-padded batch
selects a position inside the padding and returns whatever that slot happens to hold.
The gather itself is written with `take_along_axis` over an index broadcast to
`(B, 1, D)` because the index rank must match the tensor's, not because a per-row
scalar would be conceptually different.

Two optional post-processing steps follow. `truncate_dim` slices the leading
components of the vector, which is only meaningful under Matryoshka training — the
property that a prefix of the vector is itself a usable embedding is created by the
loss, not by the slice, so truncating an ordinarily-trained model degrades quality
arbitrarily. L2 normalization projects to the unit sphere, after which inner product
and cosine similarity coincide and a maximum-inner-product index answers cosine
queries exactly. Note the order: truncation happens *before* normalization, so a
truncated vector is a unit vector in its own subspace rather than a slice of a unit
vector, which is the behaviour a downstream cosine search expects.

The reranker scores a single prompt containing instruction, query and document, and
converts the model's own next-token distribution into a probability. It reads the
logits at the last non-padding position, extracts the two entries for the "yes" and
"no" token ids, and softmaxes over just that pair:

`score = softmax([logit_no, logit_yes])[1]`

Restricting the softmax to two entries rather than the full vocabulary is what makes
the number a calibrated binary probability instead of a quantity dominated by
whatever else the model might have said. It also means the score depends on the
tokenizer: `yes_token_id` and `no_token_id` default to 9891 and 2201, and pointing
this model at a different vocabulary without changing them silently scores two
unrelated tokens.

**The two towers are deliberately masked differently, and the asymmetry is the
point.** The reranker converts its own language-modelling head into a judgment, so
it is a next-token prediction and is causal: `Qwen3RerankerLayer.call` builds the
mask through `components.build_causal_attention_mask` — the same constructor
`qwen3.py` and `qwen3_next.py` use — which OR-combines a lower-triangular mask with
the caller's padding mask and returns ATTEND semantics. Before this was wired the
head scored a position that had already attended to the tokens after it; perturbing
a token at index 6 of a 12-token sequence moved every hidden state at index < 6 by
up to 3.42e-01, and now moves them by exactly 0.0 while positions >= 6 still
respond.

`Qwen3EmbeddingLayer` stays **bidirectional on purpose** and forwards the caller's
2D padding mask unchanged. Nothing is predicted from the pooled vector, so there is
no target to leak; bidirectional attention is strictly more informative for an
encoder, and it is what the strong open embedding models do. The one thing it costs
is the usual justification for last-token pooling — in a bidirectional trunk every
position has seen the whole sequence, so reading the last real one is a convention
carried over from the decoder-only lineage rather than a requirement. It remains a
reasonable convention (that position is the only one guaranteed to exist and to be
non-padding for every row), but do not read it as evidence of causality.

Position is encoded with *learned absolute* embeddings (`positional_learned` from
the embedding factory), not the rotary embeddings used elsewhere in the Qwen family.
Sequences longer than `max_seq_len` have no encoding to draw on. The trunk's
attention, FFN and normalization types are all factory keys, so these classes
describe a configurable transformer in the shape of the published models rather than
a weight-compatible port; no pretrained weights are distributed and none of the
published evaluation numbers should be expected from them.

References:
    - Zhang et al., 2025. Qwen3 Embedding: Advancing Text Embedding and Reranking
      Through Foundation Models. (https://arxiv.org/abs/2506.05176)
    - Kusupati et al., 2022. Matryoshka Representation Learning.
      (https://arxiv.org/abs/2205.13147)
    - Wang et al., 2024. Improving Text Embeddings with Large Language Models.
      (https://arxiv.org/abs/2401.00368)
    - Nogueira et al., 2020. Document Ranking with a Pretrained Sequence-to-Sequence
      Model. (https://arxiv.org/abs/2003.06713)
    - Vaswani et al., 2017. Attention Is All You Need.
      (https://arxiv.org/abs/1706.03762)
"""

import keras
from keras import ops, layers
from typing import Optional, Tuple, Dict, Any, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.transformers import TransformerLayer
from dl_techniques.layers.embedding.factory import create_embedding_layer
from dl_techniques.layers.norms.factory import create_normalization_layer

from .components import build_causal_attention_mask

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

        # Last-token pooling: gather the hidden state at each sequence's final
        # non-padded position. take_along_axis requires the index tensor to match
        # the rank of `hidden_states` (B, T, D), so broadcast the per-row last
        # index to (B, 1, D) before gathering.
        sequence_lengths = ops.sum(ops.cast(attention_mask, "int32"), axis=1) - 1
        hidden_dim = ops.shape(hidden_states)[-1]

        gather_indices = ops.reshape(sequence_lengths, (-1, 1, 1))
        gather_indices = ops.broadcast_to(
            gather_indices, (ops.shape(hidden_states)[0], 1, hidden_dim)
        )
        pooled_embeddings = ops.take_along_axis(
            hidden_states, gather_indices, axis=1
        )
        pooled_embeddings = ops.squeeze(pooled_embeddings, axis=1)

        # Apply optional dimension truncation (MRL)
        if self.truncate_dim:
            pooled_embeddings = pooled_embeddings[:, :self.truncate_dim]

        # Apply optional L2 normalization
        if self.normalize:
            pooled_embeddings = ops.normalize(pooled_embeddings, axis=1)

        return pooled_embeddings

    def compute_output_shape(
        self,
        input_shape: Union[Dict[str, Tuple], Tuple[Optional[int], ...]]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape.

        Args:
            input_shape: Either a dict with 'input_ids' shape or a tuple.

        Returns:
            Output shape: (batch_size, embedding_dimension).
        """
        if isinstance(input_shape, dict):
            batch_size = input_shape.get('input_ids', (None, None))[0]
        else:
            batch_size = input_shape[0] if input_shape else None
        output_dim = self.truncate_dim if self.truncate_dim else self.hidden_size
        return (batch_size, output_dim)

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
    N × TransformerLayer (CAUSAL + padding mask) -> Language Modeling Head ->
    Logits["no", "yes"] -> Softmax -> Score
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

        # The reranker reads its own LM head at the last real position, so it is
        # a next-token prediction and MUST be causal: without the mask that
        # position has already attended to the tokens it is being asked to
        # score. Shared with qwen3.py / qwen3_next.py; returns ATTEND semantics
        # and folds the padding mask in, so the raw 2D mask is not forwarded.
        causal_attend_mask = build_causal_attention_mask(
            hidden_states, attention_mask
        )

        # Process through transformer layers
        for transformer_layer in self.transformer_layers:
            hidden_states = transformer_layer(
                hidden_states,
                attention_mask=causal_attend_mask,
                training=training
            )

        # Apply final normalization
        hidden_states = self.final_norm(hidden_states)

        # Get logits from language modeling head
        logits = self.lm_head(hidden_states)

        # Get the logits for the last token in each sequence. take_along_axis
        # needs the index tensor to match the rank of `logits` (B, T, V), so
        # broadcast the per-row last index to (B, 1, V) before gathering.
        sequence_lengths = ops.sum(ops.cast(attention_mask, "int32"), axis=1) - 1
        vocab_dim = ops.shape(logits)[-1]

        gather_indices = ops.reshape(sequence_lengths, (-1, 1, 1))
        gather_indices = ops.broadcast_to(
            gather_indices, (ops.shape(logits)[0], 1, vocab_dim)
        )
        last_token_logits = ops.take_along_axis(
            logits, gather_indices, axis=1
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

    def compute_output_shape(
        self,
        input_shape: Union[Dict[str, Tuple], Tuple[Optional[int], ...]]
    ) -> Tuple[Optional[int]]:
        """Compute output shape.

        Args:
            input_shape: Either a dict with 'input_ids' shape or a tuple.

        Returns:
            Output shape: (batch_size,) for relevance scores.
        """
        if isinstance(input_shape, dict):
            batch_size = input_shape.get('input_ids', (None, None))[0]
        else:
            batch_size = input_shape[0] if input_shape else None
        return (batch_size,)

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
