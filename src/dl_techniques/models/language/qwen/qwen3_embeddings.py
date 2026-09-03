"""
Qwen3-style text embedding and reranking towers: :class:`Qwen3EmbeddingModel`
and :class:`Qwen3RerankerModel`, sharing a transformer trunk read out either
as a pooled vector or as a two-token relevance judgment.

An embedding model reduces a passage to one vector before it knows the
query, so a corpus can be encoded once and searched by inner product. A
reranker instead reads the query and document together in one forward pass
and scores their match directly, at the cost of one pass per pair. The
embedding tower pools the hidden state at the last non-padding position and
attends bidirectionally, since nothing is predicted from the pooled vector.
The reranker instead reads its own next-token distribution at the last
position, restricted to the "yes"/"no" token ids, and attends causally,
since it is doing next-token prediction. Position is encoded with learned
absolute embeddings, not rotary; sequences longer than ``max_seq_len`` have
no encoding to draw on.

The trunk's attention, FFN and normalization types are all factory keys, so
these classes describe a configurable transformer in the shape of the
published models, not a weight-compatible port. No pretrained weights are
distributed and no published evaluation number should be expected from them.

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
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.qwen.qwen3_embeddings")
class Qwen3EmbeddingLayer(keras.layers.Layer):
    """
    A bidirectional transformer that pools tokens into one embedding vector.

    Architecture:

    .. code-block:: text

        input_ids, attention_mask   [B, S]
                  │
                  ▼
        token + positional embeddings
                  │
                  ▼
        N x TransformerLayer (bidirectional)
                  │
                  ▼
        last-token pool          [B, hidden_size]
                  │
                  ▼
        truncate to truncate_dim (optional, MRL)
                  │
                  ▼
        L2 normalize (optional)
                  │
                  ▼
        embedding                [B, embedding_dim]

    ``last_token_idx = sum(attention_mask) - 1`` counts real tokens rather
    than locating them, so it assumes right-padding: a left-padded batch
    selects a position inside the padding. Truncation happens before
    normalization, so a truncated vector is a unit vector in its own
    subspace, which is what a downstream cosine search expects.

    :param vocab_size: Size of the token vocabulary.
    :type vocab_size: int
    :param hidden_size: Width of the hidden representations throughout the
        model.
    :type hidden_size: int
    :param num_layers: Number of transformer layers to stack.
    :type num_layers: int
    :param num_heads: Attention heads per transformer layer.
    :type num_heads: int
    :param intermediate_size: Width of the FFN's intermediate layer.
    :type intermediate_size: int
    :param max_seq_len: Maximum sequence length the positional embedding
        supports.
    :type max_seq_len: int
    :param normalize: Whether to L2-normalize the final embedding.
    :type normalize: bool
    :param truncate_dim: If set, truncate the embedding to this many leading
        dimensions (Matryoshka Representation Learning). Meaningful only
        under Matryoshka-trained weights.
    :type truncate_dim: Optional[int]
    :param dropout_rate: Dropout rate applied throughout the model.
    :type dropout_rate: float
    :param ffn_type: FFN variant (``'mlp'``, ``'swiglu'``, ``'geglu'``, etc.).
    :type ffn_type: str
    :param normalization_type: Normalization variant (``'layer_norm'``,
        ``'rms_norm'``, etc.).
    :type normalization_type: str
    :param attention_type: Attention mechanism variant.
    :type attention_type: str
    :param kwargs: Forwarded to the base ``Layer`` class.

    Input shape:
        A dictionary containing:
        - 'input_ids': Tensor of shape `(batch_size, sequence_length)`.
        - 'attention_mask': Tensor of shape `(batch_size, sequence_length)`.

    Output shape:
        A 2D tensor of shape `(batch_size, embedding_dimension)`, where
        `embedding_dimension` is `hidden_size` or `truncate_dim` if specified.

    Example:
        .. code-block:: python

            embedding_layer = Qwen3EmbeddingLayer(
                vocab_size=32000,
                hidden_size=1024,
                num_layers=12,
                num_heads=16,
                intermediate_size=2816,
                max_seq_len=8192,
                truncate_dim=256,
            )
            inputs = {
                'input_ids': tf.constant([[1, 2, 3, 4, 0]]),
                'attention_mask': tf.constant([[1, 1, 1, 1, 0]]),
            }
            embeddings = embedding_layer(inputs)  # shape (1, 256)
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

        # DECISION plan-2026-08-18T140459-7991552f/D-020: forward dropout_rate
        # here; do not add a second standalone Dropout layer -- stacking both gives an effective rate of 1-(1-p)^2. See decisions.md.
        self.positional_embeddings = create_embedding_layer(
            'positional_learned',
            max_seq_len=max_seq_len,
            dim=hidden_size,
            dropout_rate=dropout_rate,
            name='positional_embeddings'
        )

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

        # Add positional embeddings. The embedding dropout lives INSIDE this
        # layer, so `training` has to be forwarded explicitly here.
        hidden_states = self.positional_embeddings(hidden_states, training=training)

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

@register_dl_technique("dl_techniques.models.qwen.qwen3_embeddings")
class Qwen3RerankerLayer(keras.layers.Layer):
    """
    A causal transformer that scores a formatted query-document prompt as a
    "yes"/"no" relevance judgment.

    Architecture:

    .. code-block:: text

        formatted prompt          [B, S]
                  │
                  ▼
        token + positional embeddings
                  │
                  ▼
        N x TransformerLayer (causal + padding mask)
                  │
                  ▼
        language modelling head       -> logits [B, S, vocab]
                  │
                  ▼
        logits at last position, [no_id, yes_id] only
                  │
                  ▼
        softmax                       -> score [B]

    Restricting the softmax to the two token ids, rather than the full
    vocabulary, is what makes the result a calibrated probability instead
    of a value dominated by whatever else the model might say.

    :param vocab_size: Size of the vocabulary.
    :type vocab_size: int
    :param hidden_size: Hidden dimension throughout the model.
    :type hidden_size: int
    :param num_layers: Number of transformer layers.
    :type num_layers: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param intermediate_size: Width of the FFN's intermediate layer.
    :type intermediate_size: int
    :param max_seq_len: Maximum sequence length.
    :type max_seq_len: int
    :param dropout_rate: Dropout rate applied throughout the model.
    :type dropout_rate: float
    :param ffn_type: FFN variant.
    :type ffn_type: str
    :param normalization_type: Normalization variant.
    :type normalization_type: str
    :param attention_type: Attention mechanism variant.
    :type attention_type: str
    :param yes_token_id: Vocabulary id of the "yes" token.
    :type yes_token_id: int
    :param no_token_id: Vocabulary id of the "no" token.
    :type no_token_id: int
    :param kwargs: Forwarded to the base ``Layer`` class.

    Input shape:
        A dictionary containing:
        - 'input_ids': Tensor of shape `(batch_size, sequence_length)`.
        - 'attention_mask': Tensor of shape `(batch_size, sequence_length)`.

    Output shape:
        A 1D tensor of shape `(batch_size,)` containing relevance scores
        between 0 and 1.

    Note:
        Attention here is causal, built by
        :func:`.components.build_causal_attention_mask`: this layer scores
        its own next-token prediction, so a position must not attend to
        tokens after it. :class:`Qwen3EmbeddingLayer` attends bidirectionally
        instead, since it predicts nothing from the pooled vector.

    Example:
        .. code-block:: python

            reranker_layer = Qwen3RerankerLayer(
                vocab_size=32000,
                hidden_size=1024,
                num_layers=12,
                num_heads=16,
                yes_token_id=9891,
                no_token_id=2201,
            )
            inputs = {
                'input_ids': tf.constant([[1, 2, 3, 4, 5]]),
                'attention_mask': tf.constant([[1, 1, 1, 1, 1]]),
            }
            scores = reranker_layer(inputs)  # shape (1,)
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

        # DECISION plan-2026-08-18T140459-7991552f/D-020: forward dropout_rate
        # here; do not add a second standalone Dropout layer -- stacking both gives an effective rate of 1-(1-p)^2. See decisions.md.
        self.positional_embeddings = create_embedding_layer(
            'positional_learned',
            max_seq_len=max_seq_len,
            dim=hidden_size,
            dropout_rate=dropout_rate,
            name='positional_embeddings'
        )

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

        # Add positional embeddings. The embedding dropout lives INSIDE this
        # layer, so `training` has to be forwarded explicitly here.
        hidden_states = self.positional_embeddings(hidden_states, training=training)

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

@register_dl_technique("dl_techniques.models.qwen.qwen3_embeddings")
class Qwen3EmbeddingModel(keras.Model):
    """
    A `compile()`- and `fit()`-ready Keras Model wrapping :class:`Qwen3EmbeddingLayer`.

    Defines `__init__`, `call` and `get_config` and nothing else: no
    query/document helper methods, no instruction handling, no tokenizer.
    Callers pass already-tokenized `input_ids`/`attention_mask` and prefix
    any task instruction into the token ids themselves.

    :param vocab_size: Size of the token vocabulary.
    :type vocab_size: int
    :param hidden_size: Width of the hidden representations.
    :type hidden_size: int
    :param num_layers: Number of transformer layers.
    :type num_layers: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param intermediate_size: Width of the FFN's intermediate layer.
    :type intermediate_size: int
    :param max_seq_len: Maximum sequence length.
    :type max_seq_len: int
    :param normalize: Whether to L2-normalize embeddings.
    :type normalize: bool
    :param truncate_dim: Optional dimension for Matryoshka Representation
        Learning.
    :type truncate_dim: Optional[int]
    :param dropout_rate: Dropout rate throughout the model.
    :type dropout_rate: float
    :param ffn_type: FFN variant.
    :type ffn_type: str
    :param normalization_type: Normalization variant.
    :type normalization_type: str
    :param attention_type: Attention mechanism variant.
    :type attention_type: str
    :param kwargs: Forwarded to the base ``Model`` class.

    Example:
        .. code-block:: python

            model = Qwen3EmbeddingModel(
                vocab_size=32000,
                hidden_size=1024,
                num_layers=12,
                num_heads=16,
                truncate_dim=256,
            )
            inputs = {
                'input_ids': tf.constant([[1, 2, 3, 4, 0]]),
                'attention_mask': tf.constant([[1, 1, 1, 1, 0]]),
            }
            embeddings = model(inputs)  # shape (1, 256)
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

@register_dl_technique("dl_techniques.models.qwen.qwen3_embeddings")
class Qwen3RerankerModel(keras.Model):
    """
    A `compile()`- and `fit()`-ready Keras Model wrapping :class:`Qwen3RerankerLayer`.

    Defines `__init__`, `call` and `get_config` and nothing else: no prompt
    formatter, no method for processing query/document pairs. The caller
    builds the "yes"/"no" prompt and tokenizes it, then passes
    `input_ids`/`attention_mask`.

    :param vocab_size: Size of the vocabulary.
    :type vocab_size: int
    :param hidden_size: Hidden dimension throughout the model.
    :type hidden_size: int
    :param num_layers: Number of transformer layers.
    :type num_layers: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param intermediate_size: Width of the FFN's intermediate layer.
    :type intermediate_size: int
    :param max_seq_len: Maximum sequence length.
    :type max_seq_len: int
    :param dropout_rate: Dropout rate throughout the model.
    :type dropout_rate: float
    :param ffn_type: FFN variant.
    :type ffn_type: str
    :param normalization_type: Normalization variant.
    :type normalization_type: str
    :param attention_type: Attention mechanism variant.
    :type attention_type: str
    :param yes_token_id: Vocabulary id of the "yes" token.
    :type yes_token_id: int
    :param no_token_id: Vocabulary id of the "no" token.
    :type no_token_id: int
    :param kwargs: Forwarded to the base ``Model`` class.

    Example:
        .. code-block:: python

            reranker = Qwen3RerankerModel(
                vocab_size=32000,
                hidden_size=1024,
                num_layers=12,
                num_heads=16,
                yes_token_id=9891,
                no_token_id=2201,
            )
            inputs = {
                'input_ids': tf.constant([[1, 2, 3, 4, 5]]),
                'attention_mask': tf.constant([[1, 1, 1, 1, 1]]),
            }
            scores = reranker(inputs)  # shape (1,)
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
