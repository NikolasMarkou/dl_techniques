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
from keras import ops, initializers, regularizers
from typing import Optional, Union, Tuple, Dict, Any, Literal, List

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from .attention_pooling import AttentionPooling
from .weighted_pooling import WeightedPooling

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

    Args:
        strategy: Pooling strategy name or list of strategy names.
        exclude_positions: Positions to exclude from pooling.
        aggregation_method: How to combine multiple strategy outputs.
        attention_hidden_dim: Hidden dimension for attention pooling.
        attention_num_heads: Number of heads for multi-head attention.
        attention_dropout: Dropout rate for attention mechanisms.
        weighted_max_seq_len: Maximum sequence length for weighted pooling.
        top_k: Number of top elements for top-k pooling.
        temperature: Temperature for softmax in learnable strategies.
        use_bias: Whether to use bias in learnable components.
        kernel_initializer: Initializer for kernels.
        bias_initializer: Initializer for biases.
        kernel_regularizer: Optional regularizer for kernels.
        bias_regularizer: Optional regularizer for biases.
    """

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

        Args:
            input_shape: Shape tuple of the input tensor.
        """
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

        Args:
            inputs: Input tensor.
            mask: Optional boolean mask.

        Returns:
            Tuple of (masked inputs, updated mask).
        """
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

        Args:
            strategy: Name of the pooling strategy.
            inputs: Input sequence tensor.
            mask: Optional boolean mask.
            training: Whether in training mode.

        Returns:
            Pooled tensor.
        """
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

        Args:
            inputs: Input tensor ``(batch, seq_len, hidden_dim)``.
            mask: Optional boolean mask ``(batch, seq_len)``.
            training: Whether in training mode.

        Returns:
            Pooled features tensor.
        """
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

        Args:
            input_shape: Shape tuple of the input.

        Returns:
            Output shape tuple.
        """
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

        Returns:
            Dictionary containing all constructor parameters.
        """
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