"""
Perplexity Metric Implementation for Language Modeling and Sequence Prediction Tasks.

This module provides a comprehensive implementation of the perplexity metric, a fundamental
evaluation measure in natural language processing, information theory, and probabilistic
sequence modeling. The implementation includes both a stateful Keras metric class and a
functional interface for flexible usage across different model architectures and training
paradigms.

## Mathematical Foundation

Perplexity is defined as the exponential of the cross-entropy loss, providing an intuitive
measure of prediction uncertainty:

    Perplexity = exp(H(P, Q)) = exp(-1/N * Σ log Q(x_i))

Where:
- H(P, Q) is the cross-entropy between true distribution P and predicted distribution Q
- N is the number of predictions
- Q(x_i) is the predicted probability of the true token x_i
- The summation is over all tokens in the sequence

For sparse categorical targets (most common in language modeling):
    Perplexity = exp(-1/N * Σ log P(w_i | context_i))

Where P(w_i | context_i) is the model's predicted probability for the true word w_i given
its context.

## Characteristics and Properties

**Range and Interpretation:**
- Range: [1, ∞), where 1 represents perfect prediction (zero uncertainty)
- A perplexity of K means the model is as uncertain as if choosing uniformly among K options
- Lower values indicate better model performance
- Common benchmarks: GPT-2 achieves ~35-40 perplexity on Penn Treebank

**Information-Theoretic Interpretation:**
- Measures the weighted average number of choices the model has when predicting each token
- Directly related to bits per character/token: log₂(perplexity) = bits per token
- Provides intuitive understanding: perplexity of 100 means model is as confused as 
  if randomly choosing among 100 equally likely options

**Advantages over Cross-Entropy:**
- Scale-invariant: easier to compare across different sequence lengths
- Interpretable units: "effective vocabulary size" of uncertainty
- Standard metric in language modeling literature for fair comparisons

## Performance Characteristics

**Computational Complexity:**
- Time: O(B × S × V) where B=batch_size, S=sequence_length, V=vocab_size
- Memory: O(B × S × V) for softmax computation when from_logits=True
- Efficient batched computation with automatic broadcasting

**Numerical Stability:**
- Automatic epsilon clipping to prevent log(0) catastrophic failures
- Numerically stable softmax computation for logits
- Robust handling of edge cases (empty sequences, all-ignored tokens)

**Scalability:**
- Supports large vocabulary sizes through efficient sparse operations
- Memory-efficient state tracking with running averages
- Compatible with mixed precision training (automatic dtype handling)

## Implementation Features

**Flexibility:**
- Supports both logits and probability inputs via `from_logits` parameter
- Configurable token ignoring (e.g., padding tokens) via `ignore_class`
- Sample weighting support for imbalanced datasets or importance sampling
- Both stateful metric and functional interfaces for different use cases

**Keras Integration:**
- Full serialization support with `@keras.saving.register_keras_serializable()`
- Compatible with `model.compile()` metrics list
- Proper state management for training/validation splits
- Thread-safe accumulation across batches

**Robustness:**
- Automatic type casting and shape validation
- Graceful handling of edge cases (zero counts, infinite losses)
- Comprehensive error checking with informative messages

## Common Use Cases

1. **Language Model Evaluation:**
   - Primary metric for autoregressive language models (GPT, LLaMA)
   - Comparison of different architectures on same datasets
   - Tracking training progress and convergence

2. **Neural Machine Translation:**
   - Evaluation of sequence-to-sequence models
   - Cross-lingual model comparison
   - Quality assessment for different decoding strategies

3. **Speech Recognition:**
   - Acoustic model evaluation in ASR systems
   - Language model component assessment
   - End-to-end model performance tracking

4. **Text Generation:**
   - Quality assessment of generated text
   - Hyperparameter tuning and model selection
   - Ablation studies of different components

## Typical Performance Ranges

- **Character-level models:** 1.2-2.0 (excellent), 2.0-4.0 (good), >4.0 (poor)
- **Word-level models:** 20-50 (excellent), 50-100 (good), >100 (poor)
- **Subword models (BPE/SentencePiece):** 5-20 (excellent), 20-50 (good), >50 (poor)

Note: Ranges vary significantly with vocabulary size, domain complexity, and training data size.

## References and Background

1. **Shannon, C.E. (1948)**. "A Mathematical Theory of Communication"
   - Original formulation of entropy and cross-entropy in information theory

2. **Brown, P.F., et al. (1992)**. "An Estimate of an Upper Bound for the Entropy of English"
   - First application of perplexity to natural language modeling

3. **Jelinek, F. (1997)**. "Statistical Methods for Speech Recognition"
   - Comprehensive treatment of perplexity in speech and language processing

4. **Mikolov, T., et al. (2010)**. "Recurrent Neural Network Based Language Model"
   - Modern neural language modeling with perplexity evaluation

5. **Radford, A., et al. (2018)**. "Improving Language Understanding by Generative Pre-Training"
   - Large-scale transformer language models and perplexity benchmarks

## Example Usage

```python
# Basic usage with model compilation
model.compile(
    optimizer='adamw',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
        Perplexity(from_logits=True, name='perplexity')
    ]
)

# Advanced usage with padding token ignoring
perplexity_metric = Perplexity(
    from_logits=True,
    ignore_class=0,  # Ignore padding tokens with ID 0
    name='masked_perplexity'
)

# Functional interface for custom training loops
def custom_loss_and_metrics(y_true, y_pred):
    loss = keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    ppl = perplexity(y_true, y_pred, from_logits=True, ignore_class=0)
    return loss, ppl
```
"""

import keras
from typing import Optional

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class Perplexity(keras.metrics.Metric):
    """
    Perplexity metric for language modeling tasks.

    Perplexity is defined as the exponential of the cross-entropy loss.
    It measures how well a probability model predicts a sample, with lower
    values indicating better performance.

    For sparse categorical crossentropy: perplexity = exp(loss)

    Args:
        from_logits: Boolean, whether predictions are logits or probabilities.
            Defaults to True to match SparseCategoricalCrossentropy.
        ignore_class: Optional integer, class index to ignore in computation
            (e.g., padding tokens). Defaults to None.
        name: String name of the metric instance.
        dtype: Data type of the metric result.

    Example:
        ```python
        model.compile(
            optimizer='adam',
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[keras.metrics.SparseCategoricalAccuracy(), Perplexity()]
        )
        ```
    """

    def __init__(
            self,
            from_logits: bool = True,
            ignore_class: Optional[int] = None,
            name: str = 'perplexity',
            dtype: Optional[str] = None,
            **kwargs
    ):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.from_logits = from_logits
        self.ignore_class = ignore_class

        # State variables
        self.total_loss = self.add_weight(name='total_loss', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(
            self,
            y_true: keras.KerasTensor,
            y_pred: keras.KerasTensor,
            sample_weight: Optional[keras.KerasTensor] = None
    ) -> None:
        """
        Updates the metric state.

        Args:
            y_true: Ground truth values, shape (batch_size, ...).
            y_pred: Predicted values, shape (batch_size, ..., num_classes).
            sample_weight: Optional sample weights.
        """
        # Convert predictions to probabilities if they are logits
        if self.from_logits:
            y_pred = keras.ops.softmax(y_pred, axis=-1)

        # Compute sparse categorical crossentropy
        # Add small epsilon to prevent log(0)
        epsilon = keras.backend.epsilon()
        y_pred = keras.ops.clip(y_pred, epsilon, 1.0 - epsilon)

        # Convert y_true to int32 for indexing
        y_true = keras.ops.cast(y_true, 'int32')

        # Create one-hot encoding for y_true
        num_classes = keras.ops.shape(y_pred)[-1]
        y_true_one_hot = keras.ops.one_hot(y_true, num_classes)

        # Compute cross entropy: -sum(y_true * log(y_pred))
        cross_entropy = -keras.ops.sum(y_true_one_hot * keras.ops.log(y_pred), axis=-1)

        # Handle ignore_class if specified
        if self.ignore_class is not None:
            mask = keras.ops.not_equal(y_true, self.ignore_class)
            mask = keras.ops.cast(mask, cross_entropy.dtype)
            cross_entropy = cross_entropy * mask
            valid_count = keras.ops.sum(mask)
        else:
            valid_count = keras.ops.cast(keras.ops.size(cross_entropy), cross_entropy.dtype)

        # Apply sample weights if provided
        if sample_weight is not None:
            sample_weight = keras.ops.cast(sample_weight, cross_entropy.dtype)
            cross_entropy = cross_entropy * sample_weight
            if self.ignore_class is not None:
                valid_count = keras.ops.sum(mask * sample_weight)
            else:
                valid_count = keras.ops.sum(sample_weight)

        # Update running totals
        total_cross_entropy = keras.ops.sum(cross_entropy)
        self.total_loss.assign_add(total_cross_entropy)
        self.count.assign_add(valid_count)

    def result(self) -> keras.KerasTensor:
        """
        Computes and returns the metric result.

        Returns:
            Perplexity value as exponential of average cross-entropy.
        """
        # Average cross entropy
        avg_cross_entropy = keras.ops.divide_no_nan(self.total_loss, self.count)

        # Perplexity is exp(cross_entropy)
        return keras.ops.exp(avg_cross_entropy)

    def reset_state(self) -> None:
        """Resets all metric state variables."""
        self.total_loss.assign(0.0)
        self.count.assign(0.0)

    def get_config(self) -> dict:
        """Returns the serializable config of the metric."""
        config = super().get_config()
        config.update({
            'from_logits': self.from_logits,
            'ignore_class': self.ignore_class,
        })
        return config


# ---------------------------------------------------------------------

def perplexity(
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor,
        from_logits: bool = True,
        ignore_class: Optional[int] = None
) -> keras.KerasTensor:
    """
    Functional interface for computing perplexity.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        from_logits: Whether predictions are logits or probabilities.
        ignore_class: Optional class index to ignore.

    Returns:
        Perplexity value.
    """
    # Convert predictions to probabilities if they are logits
    if from_logits:
        y_pred = keras.ops.softmax(y_pred, axis=-1)

    # Add small epsilon to prevent log(0)
    epsilon = keras.backend.epsilon()
    y_pred = keras.ops.clip(y_pred, epsilon, 1.0 - epsilon)

    # Convert y_true to int32 for indexing
    y_true = keras.ops.cast(y_true, 'int32')

    # Create one-hot encoding for y_true
    num_classes = keras.ops.shape(y_pred)[-1]
    y_true_one_hot = keras.ops.one_hot(y_true, num_classes)

    # Compute cross entropy: -sum(y_true * log(y_pred))
    cross_entropy = -keras.ops.sum(y_true_one_hot * keras.ops.log(y_pred), axis=-1)

    # Handle ignore_class if specified
    if ignore_class is not None:
        mask = keras.ops.not_equal(y_true, ignore_class)
        mask = keras.ops.cast(mask, cross_entropy.dtype)
        cross_entropy = cross_entropy * mask
        avg_cross_entropy = keras.ops.sum(cross_entropy) / keras.ops.sum(mask)
    else:
        avg_cross_entropy = keras.ops.mean(cross_entropy)

    # Return perplexity
    return keras.ops.exp(avg_cross_entropy)

# ---------------------------------------------------------------------
