"""
A unified and configurable pooling layer for sequence data.

This layer serves as a bridge between sequence encoders (like Transformers or
LSTMs) and downstream tasks that require a fixed-size vector representation.
It transforms a sequence of vectors `(batch, seq_len, hidden_dim)` into a
single summary vector `(batch, output_dim)`. Its core design philosophy is
modularity, offering a wide array of pooling strategies that can be selected,
combined, and experimented with through a single, consistent interface.

Architecture:
    The layer operates on a two-stage principle: strategy execution followed by
    aggregation. First, it applies one or more user-defined pooling
    strategies to the input sequence. These strategies fall into several
    categories:
    -   **Positional:** Selects a vector from a specific position (e.g., the
        first `[CLS]` token).
    -   **Statistical:** Computes a summary statistic over the sequence
        dimension (e.g., mean, max).
    -   **Learnable:** Computes a weighted average of the sequence vectors,
        where the weights are learned during training.

    If multiple strategies are specified, their resulting vectors are combined
    in the second stage using an aggregation method, such as concatenation
    or a weighted sum, to produce the final output vector.

Foundational Mathematics and Concepts:
    The layer implements several key pooling concepts, each with a distinct
    theoretical motivation.

    1.  **Statistical and Positional Pooling:** These are simple, computationally
        efficient methods. Mean pooling (`mean`) averages all token vectors,
        capturing the overall semantic content. Max pooling (`max`) identifies
        the most salient features across the sequence, a technique popularized
        by early CNNs for NLP. Positional pooling (`cls`, `first`) relies on
        the model architecture (e.g., BERT) having learned to embed the
        summary of the entire sequence into a specific token's representation.

    2.  **Attention Pooling (`attention`):** This is a learnable, content-aware
        strategy based on the self-attention mechanism. It learns to assign
        an "importance" score to each element in the sequence and computes a
        weighted average. The process is as follows:
        -   First, each input vector `x_i` is passed through a non-linear
            transformation: `h_i = tanh(W*x_i + b)`.
        -   An unnormalized importance score `e_i` is computed by taking the
            dot product of `h_i` with a learnable context vector `u`:
            `e_i = h_i^T * u`. This context vector `u` can be interpreted as a
            learned query that represents "what is important".
        -   The scores are normalized into weights `a_i` using the softmax
            function: `a_i = softmax(e_i)`.
        -   The final representation is the weighted sum of the original input
            vectors: `v = sum(a_i * x_i)`.
        This allows the model to dynamically focus on the most relevant parts
        of the sequence for a given task.

    3.  **Weighted Pooling (`weighted`):** This provides a simpler, content-
        agnostic learnable pooling. It assigns a learnable scalar weight `p_i`
        to each *position* `i` in the sequence, up to a maximum length. These
        weights are normalized via softmax and used to compute a weighted
        average. Unlike attention, these weights are fixed after training and
        do not depend on the input content, making this method a middle ground
        between simple mean pooling and complex attention pooling.

References:
    -   Lin, Z. et al. (2017). "A Structured Self-attentive Sentence Embedding."
        This paper introduces the self-attentive pooling mechanism that forms
        the basis for the 'attention' strategy.
    -   Conneau, A. et al. (2017). "Supervised Learning of Universal Sentence
        Representations." This work demonstrated the effectiveness of simple
        pooling strategies like max-pooling over BiLSTM outputs for creating
        high-quality sentence embeddings.
    -   Zhang, T. et al. (2020). "Pooling Revisited: Your Receptive Field is
        Suboptimal." Provides a modern analysis comparing various pooling
        methods, highlighting that the optimal strategy is task-dependent.
"""

import keras
from keras import ops, layers, initializers, regularizers
from typing import Optional, Union, Tuple, Dict, Any, Literal, List

# ---------------------------------------------------------------------
# Type definitions for enhanced type safety
# ---------------------------------------------------------------------

PoolingStrategy = Literal[
    # Positional pooling
    'cls', 'first', 'last', 'middle',
    # Statistical pooling
    'mean', 'max', 'min', 'sum',
    # Advanced statistical
    'mean_max', 'mean_std', 'mean_max_min',
    # Learnable pooling
    'attention', 'multi_head_attention', 'weighted',
    # Top-k pooling
    'top_k_mean', 'top_k_max',
    # Special
    'none', 'flatten'
]

AggregationMethod = Literal['concat', 'add', 'multiply', 'weighted_sum']


# ---------------------------------------------------------------------
# Pooling Strategy Implementations
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class AttentionPooling(keras.layers.Layer):
    """Attention-based pooling that learns to weight sequence elements.

    Each token is transformed through a ``tanh`` dense layer and scored
    against a learnable context vector, producing per-token importance
    weights via softmax. The output is the weighted sum of the original
    input tokens, optionally using multiple attention heads whose outputs
    are averaged.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────┐
        │  Input [B, seq_len, embed_dim]   │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  Dense(hidden*heads, tanh)       │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  Reshape → [B, S, heads, hidden] │
        │  Score = einsum(context_vector)  │
        │  Softmax → attention weights     │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  Weighted sum over sequence      │
        │  → Output [B, embed_dim]         │
        └──────────────────────────────────┘

    :param hidden_dim: Hidden dimension for attention computation.
    :type hidden_dim: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param dropout_rate: Dropout rate for attention weights.
    :type dropout_rate: float
    :param use_bias: Whether to use bias in attention layers.
    :type use_bias: bool
    :param temperature: Temperature for attention softmax.
    :type temperature: float
    :param kernel_initializer: Initializer for kernel weights.
    :type kernel_initializer: Union[str, initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for kernel weights.
    :type kernel_regularizer: Optional[regularizers.Regularizer]"""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 1,
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        temperature: float = 1.0,
        kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        """Initialise the attention pooling layer."""
        super().__init__(**kwargs)

        # Store all configuration
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.temperature = temperature
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer) if kernel_regularizer else None

        # Create sub-layers in __init__ (Golden Rule)
        self.attention_dense = layers.Dense(
            self.hidden_dim * self.num_heads,
            activation='tanh',
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='attention_transform'
        )

        if self.dropout_rate > 0:
            self.dropout = layers.Dropout(self.dropout_rate)
        else:
            self.dropout = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build attention layers based on input shape.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]"""
        super().build(input_shape)

        embed_dim = input_shape[-1]

        # Build sub-layers explicitly (Critical for serialization)
        self.attention_dense.build(input_shape)

        # Create context vector weight
        self.context_vector = self.add_weight(
            name='context_vector',
            shape=(self.num_heads, self.hidden_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True
        )

        # Build dropout if exists
        if self.dropout is not None:
            # Dropout expects shape (batch, seq_len, num_heads)
            dropout_shape = (input_shape[0], input_shape[1], self.num_heads)
            self.dropout.build(dropout_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply attention-based pooling.

        :param inputs: Input sequence ``(batch, seq_len, embed_dim)``.
        :type inputs: keras.KerasTensor
        :param mask: Optional boolean mask ``(batch, seq_len)``.
        :type mask: Optional[keras.KerasTensor]
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :return: Pooled output ``(batch, embed_dim)``.
        :rtype: keras.KerasTensor"""
        batch_size = ops.shape(inputs)[0]
        seq_len = ops.shape(inputs)[1]

        # Compute attention scores
        attention_hidden = self.attention_dense(inputs)

        # Reshape for multi-head attention
        attention_hidden = ops.reshape(
            attention_hidden,
            (batch_size, seq_len, self.num_heads, self.hidden_dim)
        )

        # Compute attention scores with context vector
        scores = ops.einsum('bsnh,nh->bsn', attention_hidden, self.context_vector)
        scores = scores / self.temperature

        # Apply mask if provided
        if mask is not None:
            mask_expanded = ops.expand_dims(ops.cast(mask, scores.dtype), -1)
            scores = scores + (1.0 - mask_expanded) * (-1e9)

        # Compute attention weights
        attention_weights = ops.softmax(scores, axis=1)

        # Apply dropout to attention weights
        if self.dropout is not None:
            attention_weights = self.dropout(attention_weights, training=training)

        # Apply attention weights to input
        weighted_sum = ops.einsum('bsn,bsd->bnd', attention_weights, inputs)

        # Average or concatenate heads
        if self.num_heads == 1:
            output = weighted_sum[:, 0, :]
        else:
            # Average across heads
            output = ops.mean(weighted_sum, axis=1)

        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape.

        :param input_shape: Shape tuple of the input.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]"""
        return (input_shape[0], input_shape[-1])

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Dictionary containing all constructor parameters.
        :rtype: Dict[str, Any]"""
        config = super().get_config()
        config.update({
            'hidden_dim': self.hidden_dim,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate,
            'use_bias': self.use_bias,
            'temperature': self.temperature,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class WeightedPooling(keras.layers.Layer):
    """Learnable position-weighted pooling for sequences.

    A scalar learnable weight is assigned to each position up to
    ``max_seq_len``. At inference the weights for the current sequence
    length are softmax-normalised and used to compute a weighted sum,
    producing a fixed-size vector. Unlike attention pooling these weights
    are content-independent: they capture positional importance patterns
    rather than input-dependent relevance.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────┐
        │  Input [B, seq_len, embed_dim]   │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  position_weights[:seq_len]      │
        │  / temperature → softmax         │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  Weighted sum over sequence      │
        │  → Output [B, embed_dim]         │
        └──────────────────────────────────┘

    :param max_seq_len: Maximum sequence length for weight allocation.
    :type max_seq_len: int
    :param dropout_rate: Dropout rate applied to normalised weights.
    :type dropout_rate: float
    :param temperature: Temperature for weight softmax.
    :type temperature: float
    :param initializer: Initializer for position weights.
    :type initializer: Union[str, initializers.Initializer]
    :param regularizer: Optional regularizer for weights.
    :type regularizer: Optional[regularizers.Regularizer]"""

    def __init__(
        self,
        max_seq_len: int = 512,
        dropout_rate: float = 0.0,
        temperature: float = 1.0,
        initializer: Union[str, initializers.Initializer] = 'ones',
        regularizer: Optional[regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        """Initialise the weighted pooling layer."""
        super().__init__(**kwargs)

        # Store all configuration
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout_rate
        self.temperature = temperature
        self.initializer = initializers.get(initializer)
        self.regularizer = regularizers.get(regularizer) if regularizer else None

        # Create dropout layer in __init__
        if self.dropout_rate > 0:
            self.dropout = layers.Dropout(self.dropout_rate)
        else:
            self.dropout = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build learnable position weights.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]"""
        super().build(input_shape)

        # Create learnable position weights
        self.position_weights = self.add_weight(
            name='position_weights',
            shape=(self.max_seq_len,),
            initializer=self.initializer,
            regularizer=self.regularizer,
            trainable=True
        )

        # Build dropout if exists
        if self.dropout is not None:
            # Dropout expects shape (batch, seq_len)
            dropout_shape = (input_shape[0], input_shape[1])
            self.dropout.build(dropout_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply position-weighted pooling.

        :param inputs: Input sequence ``(batch, seq_len, embed_dim)``.
        :type inputs: keras.KerasTensor
        :param mask: Optional boolean mask ``(batch, seq_len)``.
        :type mask: Optional[keras.KerasTensor]
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :return: Pooled output ``(batch, embed_dim)``.
        :rtype: keras.KerasTensor"""
        seq_len = ops.shape(inputs)[1]

        # Get weights for current sequence length
        weights = self.position_weights[:seq_len]
        weights = weights / self.temperature

        # Apply mask if provided
        if mask is not None:
            weights = weights * ops.cast(mask, weights.dtype)

        # Normalize weights
        weights = ops.softmax(weights, axis=-1)

        # Apply dropout
        if self.dropout is not None:
            weights = self.dropout(weights, training=training)

        # Compute weighted sum
        weights_expanded = ops.expand_dims(weights, -1)
        weighted_sum = ops.sum(inputs * weights_expanded, axis=1)

        return weighted_sum

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape.

        :param input_shape: Shape tuple of the input.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]"""
        return (input_shape[0], input_shape[-1])

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Dictionary containing all constructor parameters.
        :rtype: Dict[str, Any]"""
        config = super().get_config()
        config.update({
            'max_seq_len': self.max_seq_len,
            'dropout_rate': self.dropout_rate,
            'temperature': self.temperature,
            'initializer': initializers.serialize(self.initializer),
            'regularizer': regularizers.serialize(self.regularizer),
        })
        return config


# ---------------------------------------------------------------------
# Main Configurable Pooling Layer
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class SequencePooling(keras.layers.Layer):
    """Configurable pooling layer supporting multiple strategies for sequences.

    This layer provides a unified interface for positional, statistical,
    learnable, and top-k pooling strategies. One or more strategies can
    be applied simultaneously and their results aggregated via
    concatenation, addition, multiplication, or learned weighted sum.
    Supported strategies include ``cls``, ``first``, ``last``, ``middle``
    (positional); ``mean``, ``max``, ``min``, ``sum`` (statistical);
    ``mean_max``, ``mean_std``, ``mean_max_min`` (combined);
    ``attention``, ``multi_head_attention``, ``weighted`` (learnable);
    ``top_k_mean``, ``top_k_max`` (top-k); and ``none``, ``flatten``
    (special).

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────┐
        │  Input [B, seq_len, hidden_dim]  │
        └──────────────┬───────────────────┘
                       │
            ┌──────────┼──────────┐
            ▼          ▼          ▼
        ┌────────┐ ┌────────┐ ┌────────┐
        │Strat 1 │ │Strat 2 │ │Strat N │
        └───┬────┘ └───┬────┘ └───┬────┘
            │          │          │
            └──────────┼──────────┘
                       ▼
        ┌──────────────────────────────────┐
        │  Aggregation                     │
        │  (concat/add/multiply/weighted)  │
        └──────────────┬───────────────────┘
                       ▼
        ┌──────────────────────────────────┐
        │  Output [B, output_dim]          │
        └──────────────────────────────────┘

    :param strategy: Pooling strategy name or list of strategy names.
    :type strategy: Union[PoolingStrategy, List[PoolingStrategy]]
    :param exclude_positions: Positions to exclude from pooling.
    :type exclude_positions: Optional[List[int]]
    :param aggregation_method: How to combine multiple strategy outputs.
    :type aggregation_method: AggregationMethod
    :param attention_hidden_dim: Hidden dimension for attention pooling.
    :type attention_hidden_dim: int
    :param attention_num_heads: Number of heads for multi-head attention.
    :type attention_num_heads: int
    :param attention_dropout: Dropout rate for attention mechanisms.
    :type attention_dropout: float
    :param weighted_max_seq_len: Maximum sequence length for weighted
        pooling.
    :type weighted_max_seq_len: int
    :param top_k: Number of top elements for top-k pooling.
    :type top_k: int
    :param temperature: Temperature for softmax in learnable strategies.
    :type temperature: float
    :param use_bias: Whether to use bias in learnable components.
    :type use_bias: bool
    :param kernel_initializer: Initializer for kernels.
    :type kernel_initializer: Union[str, initializers.Initializer]
    :param bias_initializer: Initializer for biases.
    :type bias_initializer: Union[str, initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for kernels.
    :type kernel_regularizer: Optional[regularizers.Regularizer]
    :param bias_regularizer: Optional regularizer for biases.
    :type bias_regularizer: Optional[regularizers.Regularizer]"""

    def __init__(
        self,
        strategy: Union[PoolingStrategy, List[PoolingStrategy]] = 'mean',
        exclude_positions: Optional[List[int]] = None,
        aggregation_method: AggregationMethod = 'concat',
        attention_hidden_dim: int = 256,
        attention_num_heads: int = 1,
        attention_dropout: float = 0.0,
        weighted_max_seq_len: int = 512,
        top_k: int = 10,
        temperature: float = 1.0,
        use_bias: bool = True,
        kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
        bias_initializer: Union[str, initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        bias_regularizer: Optional[regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        """Initialise the sequence pooling layer."""
        super().__init__(**kwargs)

        # Store ALL configuration (critical for get_config)
        self.strategy = strategy if isinstance(strategy, list) else [strategy]
        self.exclude_positions = exclude_positions or []
        self.aggregation_method = aggregation_method
        self.attention_hidden_dim = attention_hidden_dim
        self.attention_num_heads = attention_num_heads
        self.attention_dropout = attention_dropout
        self.weighted_max_seq_len = weighted_max_seq_len
        self.top_k = top_k
        self.temperature = temperature
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer) if kernel_regularizer else None
        self.bias_regularizer = regularizers.get(bias_regularizer) if bias_regularizer else None

        # Create learnable components in __init__ (Golden Rule)
        self.learnable_components: Dict[str, keras.layers.Layer] = {}

        for strat in self.strategy:
            if strat in ['attention', 'multi_head_attention']:
                num_heads = self.attention_num_heads if strat == 'multi_head_attention' else 1
                self.learnable_components[strat] = AttentionPooling(
                    hidden_dim=self.attention_hidden_dim,
                    num_heads=num_heads,
                    dropout_rate=self.attention_dropout,
                    use_bias=self.use_bias,
                    temperature=self.temperature,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f'{strat}_pooling'
                )
            elif strat == 'weighted':
                self.learnable_components[strat] = WeightedPooling(
                    max_seq_len=self.weighted_max_seq_len,
                    dropout_rate=self.attention_dropout,
                    temperature=self.temperature,
                    initializer='ones',
                    regularizer=self.kernel_regularizer,
                    name='weighted_pooling'
                )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer and all learnable sub-layers.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]"""
        super().build(input_shape)

        # CRITICAL: Explicitly build all learnable components
        for component in self.learnable_components.values():
            component.build(input_shape)

        # Create aggregation weights for weighted sum
        if len(self.strategy) > 1 and self.aggregation_method == 'weighted_sum':
            self.aggregation_weights = self.add_weight(
                name='aggregation_weights',
                shape=(len(self.strategy),),
                initializer='ones',
                regularizer=self.kernel_regularizer,
                trainable=True
            )

    def _apply_mask_and_exclusions(
        self,
        inputs: keras.KerasTensor,
        mask: Optional[keras.KerasTensor] = None
    ) -> Tuple[keras.KerasTensor, Optional[keras.KerasTensor]]:
        """Apply mask and position exclusions.

        :param inputs: Input tensor.
        :type inputs: keras.KerasTensor
        :param mask: Optional boolean mask.
        :type mask: Optional[keras.KerasTensor]
        :return: Tuple of (masked inputs, updated mask).
        :rtype: Tuple[keras.KerasTensor, Optional[keras.KerasTensor]]"""
        if self.exclude_positions:
            seq_len = ops.shape(inputs)[1]
            if mask is None:
                mask = ops.ones((ops.shape(inputs)[0], seq_len))

            # Create exclusion mask
            for pos in self.exclude_positions:
                if pos < seq_len:
                    indices = ops.arange(seq_len)
                    exclusion = ops.cast(indices != pos, mask.dtype)
                    mask = mask * exclusion

        return inputs, mask

    def _apply_single_strategy(
        self,
        strategy: str,
        inputs: keras.KerasTensor,
        mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply a single pooling strategy.

        :param strategy: Name of the pooling strategy.
        :type strategy: str
        :param inputs: Input sequence tensor.
        :type inputs: keras.KerasTensor
        :param mask: Optional boolean mask.
        :type mask: Optional[keras.KerasTensor]
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :return: Pooled tensor.
        :rtype: keras.KerasTensor"""
        batch_size = ops.shape(inputs)[0]
        seq_len = ops.shape(inputs)[1]

        # Apply mask and exclusions for statistical pooling
        if strategy not in ['cls', 'first', 'last', 'middle', 'none', 'flatten']:
            inputs, mask = self._apply_mask_and_exclusions(inputs, mask)

        # Positional strategies
        if strategy in ['cls', 'first']:
            return inputs[:, 0, :]

        elif strategy == 'last':
            if mask is not None:
                seq_lens = ops.sum(ops.cast(mask, 'int32'), axis=1) - 1
                seq_lens = ops.maximum(seq_lens, 0)
                batch_indices = ops.arange(batch_size)
                indices = ops.stack([batch_indices, seq_lens], axis=1)
                return ops.take_along_axis(
                    inputs,
                    ops.expand_dims(ops.expand_dims(seq_lens, -1), -1),
                    axis=1
                )[:, 0, :]
            else:
                return inputs[:, -1, :]

        elif strategy == 'middle':
            mid_pos = seq_len // 2
            return inputs[:, mid_pos, :]

        # Statistical strategies
        elif strategy == 'mean':
            if mask is not None:
                mask_expanded = ops.expand_dims(ops.cast(mask, inputs.dtype), -1)
                masked_inputs = inputs * mask_expanded
                sum_pooled = ops.sum(masked_inputs, axis=1)
                lengths = ops.sum(mask_expanded, axis=1)
                lengths = ops.maximum(lengths, 1.0)
                return sum_pooled / lengths
            else:
                return ops.mean(inputs, axis=1)

        elif strategy == 'max':
            if mask is not None:
                mask_expanded = ops.expand_dims(ops.cast(mask, inputs.dtype), -1)
                masked_inputs = inputs + (1.0 - mask_expanded) * (-1e9)
                return ops.max(masked_inputs, axis=1)
            else:
                return ops.max(inputs, axis=1)

        elif strategy == 'min':
            if mask is not None:
                mask_expanded = ops.expand_dims(ops.cast(mask, inputs.dtype), -1)
                masked_inputs = inputs + (1.0 - mask_expanded) * 1e9
                return ops.min(masked_inputs, axis=1)
            else:
                return ops.min(inputs, axis=1)

        elif strategy == 'sum':
            if mask is not None:
                mask_expanded = ops.expand_dims(ops.cast(mask, inputs.dtype), -1)
                masked_inputs = inputs * mask_expanded
                return ops.sum(masked_inputs, axis=1)
            else:
                return ops.sum(inputs, axis=1)

        # Combined statistical strategies
        elif strategy == 'mean_max':
            mean_pool = self._apply_single_strategy('mean', inputs, mask, training)
            max_pool = self._apply_single_strategy('max', inputs, mask, training)
            return ops.concatenate([mean_pool, max_pool], axis=-1)

        elif strategy == 'mean_std':
            mean_pool = self._apply_single_strategy('mean', inputs, mask, training)
            if mask is not None:
                mask_expanded = ops.expand_dims(ops.cast(mask, inputs.dtype), -1)
                masked_inputs = inputs * mask_expanded
                variance = ops.sum(
                    (masked_inputs - ops.expand_dims(mean_pool, 1)) ** 2 * mask_expanded,
                    axis=1
                )
                lengths = ops.sum(mask_expanded, axis=1)
                lengths = ops.maximum(lengths, 1.0)
                std_pool = ops.sqrt(variance / lengths + 1e-6)
            else:
                std_pool = ops.std(inputs, axis=1)
            return ops.concatenate([mean_pool, std_pool], axis=-1)

        elif strategy == 'mean_max_min':
            mean_pool = self._apply_single_strategy('mean', inputs, mask, training)
            max_pool = self._apply_single_strategy('max', inputs, mask, training)
            min_pool = self._apply_single_strategy('min', inputs, mask, training)
            return ops.concatenate([mean_pool, max_pool, min_pool], axis=-1)

        # Learnable strategies
        elif strategy in ['attention', 'multi_head_attention']:
            return self.learnable_components[strategy](
                inputs, mask=mask, training=training
            )

        elif strategy == 'weighted':
            return self.learnable_components[strategy](
                inputs, mask=mask, training=training
            )

        # Top-k strategies
        elif strategy == 'top_k_mean':
            if mask is not None:
                mask_expanded = ops.expand_dims(ops.cast(mask, inputs.dtype), -1)
                masked_inputs = inputs * mask_expanded
            else:
                masked_inputs = inputs

            norms = ops.sum(masked_inputs ** 2, axis=-1)
            k = ops.minimum(self.top_k, seq_len)
            _, top_k_indices = ops.top_k(norms, k=k)

            top_k_embeds = ops.take_along_axis(
                inputs,
                ops.expand_dims(top_k_indices, -1),
                axis=1
            )
            return ops.mean(top_k_embeds, axis=1)

        elif strategy == 'top_k_max':
            if mask is not None:
                mask_expanded = ops.expand_dims(ops.cast(mask, inputs.dtype), -1)
                masked_inputs = inputs * mask_expanded
            else:
                masked_inputs = inputs

            norms = ops.sum(masked_inputs ** 2, axis=-1)
            k = ops.minimum(self.top_k, seq_len)
            _, top_k_indices = ops.top_k(norms, k=k)

            top_k_embeds = ops.take_along_axis(
                inputs,
                ops.expand_dims(top_k_indices, -1),
                axis=1
            )
            return ops.max(top_k_embeds, axis=1)

        # Special strategies
        elif strategy == 'none':
            return inputs

        elif strategy == 'flatten':
            return ops.reshape(inputs, (batch_size, -1))

        else:
            raise ValueError(f"Unknown pooling strategy: {strategy}")

    def call(
        self,
        inputs: keras.KerasTensor,
        mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply the configured pooling strategies.

        :param inputs: Input tensor ``(batch, seq_len, hidden_dim)``.
        :type inputs: keras.KerasTensor
        :param mask: Optional boolean mask ``(batch, seq_len)``.
        :type mask: Optional[keras.KerasTensor]
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :return: Pooled features tensor.
        :rtype: keras.KerasTensor"""
        # Apply each strategy
        outputs = []
        for strat in self.strategy:
            output = self._apply_single_strategy(strat, inputs, mask, training)
            outputs.append(output)

        # Return single output if only one strategy
        if len(outputs) == 1:
            return outputs[0]

        # Handle different aggregation methods
        if self.aggregation_method == 'concat':
            if any(s == 'none' for s in self.strategy):
                raise ValueError("Cannot concatenate 'none' strategy with others")
            return ops.concatenate(outputs, axis=-1)

        elif self.aggregation_method == 'add':
            result = outputs[0]
            for output in outputs[1:]:
                result = result + output
            return result

        elif self.aggregation_method == 'multiply':
            result = outputs[0]
            for output in outputs[1:]:
                result = result * output
            return result

        elif self.aggregation_method == 'weighted_sum':
            weights = ops.softmax(self.aggregation_weights)
            result = outputs[0] * weights[0]
            for i, output in enumerate(outputs[1:], 1):
                result = result + output * weights[i]
            return result

        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape based on pooling strategy.

        :param input_shape: Shape tuple of the input.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]"""
        batch_size = input_shape[0]
        hidden_dim = input_shape[-1]

        # Handle single strategy
        if len(self.strategy) == 1:
            strat = self.strategy[0]
            if strat == 'none':
                return input_shape
            elif strat == 'flatten':
                seq_len = input_shape[1]
                return (batch_size, seq_len * hidden_dim if seq_len else None)
            elif strat in ['mean_max', 'mean_std']:
                return (batch_size, hidden_dim * 2)
            elif strat == 'mean_max_min':
                return (batch_size, hidden_dim * 3)
            else:
                return (batch_size, hidden_dim)

        # Handle multiple strategies
        if self.aggregation_method == 'concat':
            total_dim = 0
            for strat in self.strategy:
                if strat in ['mean_max', 'mean_std']:
                    total_dim += hidden_dim * 2
                elif strat == 'mean_max_min':
                    total_dim += hidden_dim * 3
                else:
                    total_dim += hidden_dim
            return (batch_size, total_dim)
        else:
            # add, multiply, weighted_sum preserve dimension
            return (batch_size, hidden_dim)

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration for serialization.

        :return: Dictionary containing all constructor parameters.
        :rtype: Dict[str, Any]"""
        config = super().get_config()
        config.update({
            'strategy': self.strategy,
            'exclude_positions': self.exclude_positions,
            'aggregation_method': self.aggregation_method,
            'attention_hidden_dim': self.attention_hidden_dim,
            'attention_num_heads': self.attention_num_heads,
            'attention_dropout': self.attention_dropout,
            'weighted_max_seq_len': self.weighted_max_seq_len,
            'top_k': self.top_k,
            'temperature': self.temperature,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------