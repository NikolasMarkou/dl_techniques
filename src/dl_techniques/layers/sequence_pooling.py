"""
Configurable Sequence Pooling Layer for Transformer Models

This module provides a highly configurable pooling layer that can be used with
any sequence-based model (text encoders, vision encoders, time series models, etc.).
It supports a wide variety of pooling strategies including simple operations like
mean/max pooling, positional selections, and advanced learnable methods like
attention-based pooling.

References:
    - Lin et al. (2017): "A Structured Self-attentive Sentence Embedding"
    - Conneau et al. (2017): "Supervised Learning of Universal Sentence Representations"
    - Zhang et al. (2020): "Pooling Revisited: Your Receptive Field is Suboptimal"
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
    """
    Attention-based pooling that learns to weight sequence elements.

    This computes attention weights for each sequence element and returns
    a weighted sum of the sequence. The attention mechanism learns what
    parts of the sequence are most important for the downstream task.

    **Intent**: Provide learnable attention-based pooling that dynamically
    weights sequence elements based on their importance, enabling the model
    to focus on relevant parts of the input sequence.

    **Architecture**:
    ```
    Input(batch, seq_len, embed_dim)
           ↓
    Dense(hidden_dim * num_heads, activation='tanh')
           ↓
    Reshape(batch, seq_len, num_heads, hidden_dim)
           ↓
    Attention Score = einsum with context_vector
           ↓
    Softmax(with masking if provided)
           ↓
    Weighted Sum = einsum with input
           ↓
    Output(batch, embed_dim)
    ```

    Args:
        hidden_dim: Hidden dimension for attention computation.
        num_heads: Number of attention heads for multi-head variant.
        dropout_rate: Dropout rate for attention weights.
        use_bias: Whether to use bias in attention layers.
        temperature: Temperature for attention softmax.
        kernel_initializer: Initializer for kernel weights.
        kernel_regularizer: Regularizer for kernel weights.

    Input shape:
        3D tensor with shape `(batch_size, sequence_length, embedding_dim)`

    Output shape:
        2D tensor with shape `(batch_size, embedding_dim)`
    """

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
        """Initialize attention pooling layer."""
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
        """Build attention layers based on input shape."""
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
        """Apply attention-based pooling."""
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
        """Compute output shape."""
        return (input_shape[0], input_shape[-1])

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
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
    """
    Learnable weighted pooling with position-specific weights.

    This learns a weight for each position in the sequence and computes
    a weighted sum. Useful when certain positions consistently contain
    more important information.

    **Intent**: Enable position-aware pooling where the model learns which
    sequence positions are most informative, providing a simpler alternative
    to attention mechanisms for position-dependent importance.

    **Architecture**:
    ```
    Input(batch, seq_len, embed_dim)
           ↓
    Get position_weights[:seq_len]
           ↓
    Apply temperature & mask
           ↓
    Softmax normalization
           ↓
    Weighted sum with input
           ↓
    Output(batch, embed_dim)
    ```

    Args:
        max_seq_len: Maximum sequence length for weight initialization.
        dropout_rate: Dropout rate for weights.
        temperature: Temperature for weight softmax.
        initializer: Weight initializer.
        regularizer: Weight regularizer.

    Input shape:
        3D tensor with shape `(batch_size, sequence_length, embedding_dim)`

    Output shape:
        2D tensor with shape `(batch_size, embedding_dim)`
    """

    def __init__(
        self,
        max_seq_len: int = 512,
        dropout_rate: float = 0.0,
        temperature: float = 1.0,
        initializer: Union[str, initializers.Initializer] = 'ones',
        regularizer: Optional[regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        """Initialize weighted pooling layer."""
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
        """Build position weights."""
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
        """Apply weighted pooling."""
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
        """Compute output shape."""
        return (input_shape[0], input_shape[-1])

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
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
    """
    Highly configurable pooling layer for sequence data.

    This layer provides a unified interface for various pooling strategies
    commonly used in sequence models like transformers. It supports both
    simple statistical pooling and advanced learnable methods.

    **Intent**: Provide a flexible, unified pooling interface that can adapt
    to different sequence modeling tasks, from simple averaging to complex
    attention-based mechanisms, enabling easy experimentation with pooling strategies.

    **Architecture**:
    ```
    Input(batch, seq_len, hidden_dim)
           ↓
    Apply Strategy 1 → Output 1
    Apply Strategy 2 → Output 2  (if multiple strategies)
    Apply Strategy N → Output N
           ↓
    Aggregation (concat/add/multiply/weighted_sum)
           ↓
    Output(batch, output_dim)
    ```

    **Pooling Strategies**:
    - **Positional**: 'cls', 'first', 'last', 'middle'
    - **Statistical**: 'mean', 'max', 'min', 'sum'
    - **Combined**: 'mean_max', 'mean_std', 'mean_max_min'
    - **Learnable**: 'attention', 'multi_head_attention', 'weighted'
    - **Top-k**: 'top_k_mean', 'top_k_max'
    - **Special**: 'none', 'flatten'

    Args:
        strategy: PoolingStrategy or list of strategies to use.
        exclude_positions: List of positions to exclude from pooling (0-indexed).
        aggregation_method: How to combine multiple strategies.
        attention_hidden_dim: Hidden dimension for attention pooling.
        attention_num_heads: Number of heads for multi-head attention pooling.
        attention_dropout: Dropout rate for attention mechanisms.
        weighted_max_seq_len: Maximum sequence length for weighted pooling.
        top_k: Number of top elements for top-k pooling.
        temperature: Temperature for softmax in attention/weighted pooling.
        use_bias: Whether to use bias in learnable components.
        kernel_initializer: Initializer for kernels.
        bias_initializer: Initializer for biases.
        kernel_regularizer: Regularizer for kernels.
        bias_regularizer: Regularizer for biases.

    Input Shape:
        3D tensor with shape `(batch_size, sequence_length, hidden_dim)`

    Output Shape:
        - For single strategies: `(batch_size, hidden_dim)` or variants
        - For 'none': `(batch_size, sequence_length, hidden_dim)`
        - For 'flatten': `(batch_size, sequence_length * hidden_dim)`

    Example:
        ```python
        # Simple mean pooling
        pooling = SequencePooling(strategy='mean')

        # Attention-based pooling
        pooling = SequencePooling(
            strategy='attention',
            attention_hidden_dim=256,
            attention_dropout=0.1
        )

        # Combination of strategies
        pooling = SequencePooling(
            strategy=['mean', 'max', 'attention'],
            aggregation_method='concat'
        )
        ```
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
        """Initialize sequence pooling layer."""
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
        """Build the layer and all sub-layers."""
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
        """Apply mask and position exclusions."""
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
        """Apply a single pooling strategy."""
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
        """
        Apply pooling strategy to inputs.

        Args:
            inputs: Input tensor of shape (batch, seq_len, hidden_dim)
            mask: Optional boolean mask of shape (batch, seq_len)
            training: Whether in training mode.

        Returns:
            Pooled features based on configured strategy.
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
        """Compute output shape based on pooling strategy."""
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
        """Get layer configuration for serialization."""
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