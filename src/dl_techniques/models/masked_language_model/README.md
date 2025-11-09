# Masked Language Model (MLM) Pre-training Framework

A production-ready, model-agnostic framework for pre-training NLP foundation models using the Masked Language Modeling (MLM) objective. Fully integrated with the `dl_techniques` framework and compatible with your BERT implementation.

## Table of Contents

1. [Overview](#overview)
2. [What is Masked Language Modeling?](#what-is-masked-language-modeling)
3. [Quick Start](#quick-start)
4. [Installation and Setup](#installation-and-setup)
5. [Pre-training Workflow](#pre-training-workflow)
6. [Configuration Guide](#configuration-guide)
7. [Working with BERT](#working-with-bert)
8. [Data Preparation](#data-preparation)
9. [Training Strategies](#training-strategies)
10. [Fine-tuning for Downstream Tasks](#fine-tuning-for-downstream-tasks)
11. [Visualization and Evaluation](#visualization-and-evaluation)
12. [Advanced Usage](#advanced-usage)
13. [Best Practices](#best-practices)
14. [Troubleshooting](#troubleshooting)
15. [API Reference](#api-reference)

---

## Overview

The Masked Language Model framework provides a self-supervised learning approach for pre-training transformer-based language models without requiring labeled data. This implementation is:

- **Model-Agnostic**: Works with any encoder implementing the standard interface
- **Framework-Compliant**: Follows all `dl_techniques` conventions
- **Production-Ready**: Comprehensive testing, serialization support, and error handling
- **Fully Documented**: Type hints and Sphinx-compatible documentation

### Key Features

- **Dynamic Masking**: BERT-style masking with 80-10-10 strategy
- **Special Token Protection**: Never masks [CLS], [SEP], [PAD], or custom special tokens
- **Efficient Loss Computation**: Loss calculated only on masked positions
- **Full Serialization**: Save and load complete models or just the encoder
- **Visualization Tools**: Inspect model predictions during training

### Why Use MLM?

- **No Labels Required**: Pre-train on vast unlabeled text corpora
- **Deep Bidirectional Context**: Learn representations using both left and right context
- **Transfer Learning**: Excellent starting point for downstream NLP tasks
- **Data Efficiency**: Requires less labeled data for fine-tuning

---

## What is Masked Language Modeling?

MLM is a self-supervised "fill-in-the-blank" task introduced by BERT:

### The Process

1. **Token Selection**: Randomly select 15% of tokens for masking
2. **Masking Strategy**: Of selected tokens:
   - 80% → Replace with `[MASK]` token
   - 10% → Replace with random token
   - 10% → Leave unchanged
3. **Encoding**: Process the corrupted sequence through the encoder
4. **Prediction**: Predict the original IDs of only the masked tokens
5. **Loss**: Compute cross-entropy loss on masked positions only

### Architecture Flow

```
Input Tokens: "the quick brown fox jumps over the lazy dog"
     ↓
Masking: "the quick brown [MASK] jumps [MASK] the lazy dog"
     ↓
Encoder (BERT/Custom) → Hidden States
     ↓
MLM Head (Dense → LayerNorm → Dense) → Logits
     ↓
Loss = CrossEntropy(original_tokens_at_masked_positions, predicted_logits)
```

This forces the model to learn rich, contextual representations by predicting masked words based on surrounding context.

---

## Quick Start

### Minimal Working Example

```python
import keras
import tensorflow as tf
from transformers import BertTokenizer
from dl_techniques.models.bert import BERT
from dl_techniques.models.masked_language_model import MaskedLanguageModel

# 1. Create encoder
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
encoder = BERT.from_variant("base", vocab_size=tokenizer.vocab_size)

# 2. Create MLM pretrainer
mlm_model = MaskedLanguageModel(
    encoder=encoder,
    vocab_size=tokenizer.vocab_size,
    mask_token_id=tokenizer.mask_token_id,
    special_token_ids=tokenizer.all_special_ids,
)

# 3. Compile
mlm_model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=5e-5, weight_decay=0.01)
)

# 4. Train (assuming train_dataset is prepared)
mlm_model.fit(train_dataset, epochs=5)

# 5. Extract pretrained encoder
pretrained_encoder = mlm_model.encoder
pretrained_encoder.save("pretrained_bert.keras")
```

---

## Pre-training Workflow

### Step 1: Prepare Your Data

Create a `tf.data.Dataset` pipeline for efficient data loading:

```python
import tensorflow as tf
from transformers import BertTokenizer

def create_mlm_dataset(
    text_file: str,
    tokenizer: BertTokenizer,
    batch_size: int = 32,
    max_length: int = 128,
    buffer_size: int = 10000,
):
    """Create an efficient dataset for MLM pre-training.
    
    Args:
        text_file: Path to text file (one sentence per line)
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size for training
        max_length: Maximum sequence length
        buffer_size: Shuffle buffer size
    """
    # Load text data
    dataset = tf.data.TextLineDataset(text_file)
    
    # Shuffle for better training
    dataset = dataset.shuffle(buffer_size)
    
    # Tokenization function
    def tokenize_function(text):
        # Tokenize using HuggingFace tokenizer
        encoded = tokenizer(
            text.numpy().decode('utf-8'),
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='np'
        )
        
        return {
            'input_ids': encoded['input_ids'][0],
            'attention_mask': encoded['attention_mask'][0],
            'token_type_ids': encoded['token_type_ids'][0],
        }
    
    # Apply tokenization
    dataset = dataset.map(
        lambda x: tf.py_function(
            tokenize_function,
            [x],
            {
                'input_ids': tf.int32,
                'attention_mask': tf.int32,
                'token_type_ids': tf.int32,
            }
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Set shapes (required after py_function)
    dataset = dataset.map(
        lambda x: {
            'input_ids': tf.ensure_shape(x['input_ids'], [max_length]),
            'attention_mask': tf.ensure_shape(x['attention_mask'], [max_length]),
            'token_type_ids': tf.ensure_shape(x['token_type_ids'], [max_length]),
        }
    )
    
    # Batch and prefetch
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# Example usage
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_dataset = create_mlm_dataset(
    "data/wiki_corpus.txt",
    tokenizer,
    batch_size=64,
    max_length=128
)
```

### Step 2: Create the MLM Model

```python
from dl_techniques.models.bert import BERT
from dl_techniques.models.masked_language_model import MaskedLanguageModel

# Create BERT encoder
encoder = BERT.from_variant(
    "base",
    vocab_size=tokenizer.vocab_size,
    max_position_embeddings=128,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
)

# Wrap with MLM pretrainer
mlm_model = MaskedLanguageModel(
    encoder=encoder,
    vocab_size=tokenizer.vocab_size,
    mask_ratio=0.15,  # 15% of tokens
    mask_token_id=tokenizer.mask_token_id,
    special_token_ids=tokenizer.all_special_ids,
    mlm_head_activation="gelu",
    mlm_head_dropout=0.1,
)
```

### Step 3: Configure Training

```python
from dl_techniques.optimization import (
    optimizer_builder,
    learning_rate_schedule_builder
)

# Calculate training steps
num_epochs = 5
steps_per_epoch = len(train_dataset)
total_steps = num_epochs * steps_per_epoch
warmup_steps = int(0.1 * total_steps)  # 10% warmup

# Configure learning rate schedule
lr_config = {
    "type": "cosine_decay",
    "learning_rate": 5e-5,
    "decay_steps": total_steps,
    "warmup_steps": warmup_steps,
    "warmup_start_lr": 1e-8,
    "alpha": 1e-6,  # Minimum LR
}

# Configure optimizer
optimizer_config = {
    "type": "adamw",
    "weight_decay": 0.01,
    "gradient_clipping_by_norm": 1.0,
}

# Build optimizer
lr_schedule = learning_rate_schedule_builder(lr_config)
optimizer = optimizer_builder(optimizer_config, lr_schedule)

# Compile model
mlm_model.compile(
    optimizer=optimizer,
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="mlm_accuracy")]
)
```

### Step 4: Train the Model

```python
# Configure callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint(
        "checkpoints/mlm_model_{epoch:02d}.keras",
        save_freq='epoch',
        save_best_only=False
    ),
    keras.callbacks.TensorBoard(
        log_dir='logs/mlm_pretraining',
        update_freq='batch'
    ),
    keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=3,
        restore_best_weights=True
    ),
]

# Start training
history = mlm_model.fit(
    train_dataset,
    epochs=num_epochs,
    callbacks=callbacks,
    verbose=1
)
```

### Step 5: Extract Pretrained Encoder

```python
# Save the encoder for downstream tasks
pretrained_encoder = mlm_model.encoder
pretrained_encoder.save("models/pretrained_bert_base.keras")

# Or save the complete MLM model (includes the head)
mlm_model.save("models/mlm_model_complete.keras")
```

---

## Configuration Guide

### MaskedLanguageModel Parameters

```python
MaskedLanguageModel(
    encoder: keras.Model,              # Required: Your encoder model
    vocab_size: int,                   # Required: Vocabulary size
    
    # Masking configuration
    mask_ratio: float = 0.15,          # Fraction of tokens to mask
    mask_token_id: int = 103,          # [MASK] token ID
    random_token_ratio: float = 0.1,   # Replace with random token
    unchanged_ratio: float = 0.1,      # Leave unchanged
    special_token_ids: List[int] = [], # Never mask these tokens
    
    # MLM head configuration
    mlm_head_activation: str = "gelu", # Activation function
    initializer_range: float = 0.02,   # Weight init stddev
    mlm_head_dropout: float = 0.1,     # Dropout rate
    layer_norm_eps: float = 1e-12,     # LayerNorm epsilon
)
```

### Recommended Configurations

#### Standard BERT Pre-training

```python
mlm_config = {
    "mask_ratio": 0.15,
    "mask_token_id": 103,
    "random_token_ratio": 0.1,
    "unchanged_ratio": 0.1,
    "mlm_head_activation": "gelu",
    "initializer_range": 0.02,
    "mlm_head_dropout": 0.1,
}
```

#### Low-Resource Setting

```python
mlm_config = {
    "mask_ratio": 0.20,  # More aggressive masking
    "random_token_ratio": 0.05,
    "unchanged_ratio": 0.05,
    "mlm_head_dropout": 0.15,  # Higher dropout for regularization
}
```

#### Domain Adaptation

```python
mlm_config = {
    "mask_ratio": 0.10,  # Gentler masking for fine-tuning
    "random_token_ratio": 0.05,
    "unchanged_ratio": 0.05,
    "mlm_head_dropout": 0.05,  # Lower dropout when adapting
}
```

---

## Working with BERT

### Using Different BERT Variants

```python
from bert import BERT

# BERT-Tiny (fast prototyping)
encoder_tiny = BERT.from_variant("tiny")  # 4 layers, 256 hidden

# BERT-Small (balanced)
encoder_small = BERT.from_variant("small")  # 6 layers, 512 hidden

# BERT-Base (standard)
encoder_base = BERT.from_variant("base")  # 12 layers, 768 hidden

# BERT-Large (maximum performance)
encoder_large = BERT.from_variant("large")  # 24 layers, 1024 hidden

# Custom BERT with advanced features
encoder_custom = BERT.from_variant(
    "base",
    normalization_type="rms_norm",  # Use RMSNorm instead of LayerNorm
    use_stochastic_depth=True,      # Enable stochastic depth
    stochastic_depth_rate=0.1,
)
```

### Extracting and Saving the Encoder

```python
# After pre-training
pretrained_encoder = mlm_model.encoder

# Save just the encoder
pretrained_encoder.save("pretrained_bert.keras")

# Load the encoder later
loaded_encoder = keras.models.load_model("pretrained_bert.keras")

# Verify it works
test_inputs = {
    "input_ids": tf.random.uniform((1, 128), 0, 30522, dtype=tf.int32),
    "attention_mask": tf.ones((1, 128), dtype=tf.int32),
}
outputs = loaded_encoder(test_inputs)
print(outputs["last_hidden_state"].shape)  # (1, 128, 768)
```

---

## Data Preparation

### Text File Format

Prepare your corpus as a text file with one sentence per line:

```text
The quick brown fox jumps over the lazy dog.
Machine learning is a subset of artificial intelligence.
Natural language processing enables computers to understand human language.
...
```

### Large-Scale Data Processing

For large corpora (e.g., Wikipedia dumps):

```python
def preprocess_wiki_dump(
    input_file: str,
    output_file: str,
    min_length: int = 10,
    max_length: int = 1000
):
    """Preprocess Wikipedia dump for MLM training."""
    import re
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for line in infile:
                # Remove wiki markup
                line = re.sub(r'<[^>]+>', '', line)
                line = line.strip()
                
                # Filter by length
                if min_length <= len(line.split()) <= max_length:
                    outfile.write(line + '\n')

# Process dump
preprocess_wiki_dump(
    'data/enwiki-latest-pages-articles.xml',
    'data/wiki_processed.txt'
)
```

### Data Augmentation

```python
def augment_text_dataset(
    dataset: tf.data.Dataset,
    augmentation_prob: float = 0.3
) -> tf.data.Dataset:
    """Apply simple data augmentation to text."""
    
    def maybe_augment(text):
        if tf.random.uniform([]) < augmentation_prob:
            # Example: Random word dropout
            words = tf.strings.split(text)
            mask = tf.random.uniform(tf.shape(words)) > 0.1
            words = tf.boolean_mask(words, mask)
            return tf.strings.reduce_join(words, separator=' ')
        return text
    
    return dataset.map(maybe_augment)
```

---

## Training Strategies

### Learning Rate Schedules

#### Warmup + Cosine Decay (Recommended)

```python
from dl_techniques.optimization import learning_rate_schedule_builder

lr_schedule = learning_rate_schedule_builder({
    "type": "cosine_decay",
    "learning_rate": 5e-5,
    "decay_steps": total_steps,
    "warmup_steps": int(0.1 * total_steps),
    "warmup_start_lr": 1e-8,
    "alpha": 1e-6,
})
```

#### Warmup + Linear Decay

```python
lr_schedule = learning_rate_schedule_builder({
    "type": "polynomial_decay",
    "learning_rate": 5e-5,
    "decay_steps": total_steps,
    "warmup_steps": int(0.1 * total_steps),
    "power": 1.0,  # Linear
})
```

### Multi-GPU Training

```python
import tensorflow as tf

# Create distribution strategy
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    # Create encoder
    encoder = BERT.from_variant("base", vocab_size=tokenizer.vocab_size)
    
    # Create MLM model
    mlm_model = MaskedLanguageModel(
        encoder=encoder,
        vocab_size=tokenizer.vocab_size,
        mask_token_id=tokenizer.mask_token_id,
    )
    
    # Scale learning rate by number of GPUs
    num_gpus = strategy.num_replicas_in_sync
    scaled_lr = 5e-5 * num_gpus
    
    # Compile
    mlm_model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=scaled_lr,
            weight_decay=0.01
        )
    )

# Train (batch will be distributed across GPUs)
mlm_model.fit(train_dataset, epochs=5)
```

### Mixed Precision Training

```python
# Enable mixed precision
keras.mixed_precision.set_global_policy('mixed_float16')

# Create model (automatically uses mixed precision)
mlm_model = MaskedLanguageModel(
    encoder=encoder,
    vocab_size=tokenizer.vocab_size,
    mask_token_id=tokenizer.mask_token_id,
)

# Compile with loss scaling
mlm_model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=5e-5),
    loss_scale='dynamic'  # Prevents underflow
)
```

### Gradient Accumulation

```python
class GradientAccumulationModel(keras.Model):
    """Wrapper for gradient accumulation."""
    
    def __init__(self, model, accumulation_steps=4, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.accumulation_steps = accumulation_steps
        self.gradient_accumulation = [
            tf.Variable(tf.zeros_like(v), trainable=False)
            for v in model.trainable_variables
        ]
    
    def train_step(self, data):
        # Implementation of gradient accumulation
        # (Simplified - full implementation would handle batching)
        pass

# Use it
mlm_with_accumulation = GradientAccumulationModel(
    mlm_model,
    accumulation_steps=4  # Effective batch size = 4x
)
```

---

## Fine-tuning for Downstream Tasks

After pre-training, use the encoder for supervised tasks:

### Text Classification

```python
# Load pretrained encoder
pretrained_encoder = keras.models.load_model("pretrained_bert.keras")
pretrained_encoder.trainable = True  # Allow fine-tuning

# Create classification model
inputs = {
    "input_ids": keras.Input(shape=(128,), dtype="int32"),
    "attention_mask": keras.Input(shape=(128,), dtype="int32"),
    "token_type_ids": keras.Input(shape=(128,), dtype="int32"),
}

# Get encoder outputs
encoder_outputs = pretrained_encoder(inputs)
cls_token = encoder_outputs["last_hidden_state"][:, 0, :]  # [CLS] token

# Add classification head
dropout = keras.layers.Dropout(0.1)(cls_token)
logits = keras.layers.Dense(num_classes)(dropout)

# Build model
classifier = keras.Model(inputs, logits)

# Compile with lower learning rate for fine-tuning
classifier.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=2e-5),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Fine-tune
classifier.fit(labeled_dataset, epochs=3)
```

### Named Entity Recognition (NER)

```python
# Token-level classification
inputs = {
    "input_ids": keras.Input(shape=(128,), dtype="int32"),
    "attention_mask": keras.Input(shape=(128,), dtype="int32"),
    "token_type_ids": keras.Input(shape=(128,), dtype="int32"),
}

# Get all token representations
encoder_outputs = pretrained_encoder(inputs)
sequence_output = encoder_outputs["last_hidden_state"]

# Token-level classification
dropout = keras.layers.Dropout(0.1)(sequence_output)
logits = keras.layers.Dense(num_entity_tags)(dropout)

# Build model
ner_model = keras.Model(inputs, logits)

# Compile and train
ner_model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=2e-5),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
```

### Using dl_techniques NLP Heads

```python
from dl_techniques.layers.nlp_heads import create_nlp_head, NLPTaskConfig, NLPTaskType

# Define task
task_config = NLPTaskConfig(
    name="sentiment",
    task_type=NLPTaskType.SENTIMENT_ANALYSIS,
    num_classes=3
)

# Create head
sentiment_head = create_nlp_head(
    task_config=task_config,
    input_dim=pretrained_encoder.hidden_size
)

# Build complete model
inputs = {
    "input_ids": keras.Input(shape=(128,), dtype="int32"),
    "attention_mask": keras.Input(shape=(128,), dtype="int32"),
}
encoder_outputs = pretrained_encoder(inputs)
task_outputs = sentiment_head(encoder_outputs)

sentiment_model = keras.Model(inputs, task_outputs)
```

---

## Visualization and Evaluation

### Visualizing MLM Predictions

```python
from mlm_pretrainer import visualize_mlm_predictions

# Get a batch from your dataset
sample_batch = next(iter(train_dataset))

# Visualize predictions
visualize_mlm_predictions(
    mlm_model=mlm_model,
    inputs=sample_batch,
    tokenizer=tokenizer,
    num_samples=4
)
```

**Example Output:**

```
================================================================================
MLM Prediction Visualization
================================================================================
--------------------------------------------------------------------------------
Sample 1
Original:     the capital of france is paris.
Masked Input: the capital of [MASK] is paris.
Prediction:   the capital of france is paris.
--------------------------------------------------------------------------------
Sample 2
Original:     machine learning is a subset of artificial intelligence.
Masked Input: machine [MASK] is a subset of [MASK] intelligence.
Prediction:   machine learning is a subset of artificial intelligence.
--------------------------------------------------------------------------------
```

### Training Metrics

```python
# Plot training curves
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

# Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['mlm_accuracy'])
plt.title('MLM Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)

plt.tight_layout()
plt.savefig('mlm_training_curves.png')
```

### Perplexity Calculation

```python
def calculate_perplexity(mlm_model, dataset, num_batches=100):
    """Calculate perplexity on a dataset."""
    total_loss = 0.0
    total_tokens = 0
    
    for i, batch in enumerate(dataset.take(num_batches)):
        # Apply masking
        masked_inputs, labels, mask = mlm_model._mask_tokens(batch)
        
        # Get predictions
        logits = mlm_model(masked_inputs, training=False)
        
        # Compute loss
        loss = mlm_model.compute_loss(
            y=labels,
            y_pred=logits,
            sample_weight=mask
        )
        
        # Accumulate
        num_masked = tf.reduce_sum(tf.cast(mask, tf.float32))
        total_loss += loss * num_masked
        total_tokens += num_masked
    
    # Calculate perplexity
    avg_loss = total_loss / total_tokens
    perplexity = tf.exp(avg_loss)
    
    return perplexity.numpy()

# Evaluate
perplexity = calculate_perplexity(mlm_model, test_dataset)
print(f"Perplexity: {perplexity:.2f}")
```

---

## Advanced Usage

### Custom Encoder Integration

Any encoder can be used as long as it follows the interface:

```python
@keras.saving.register_keras_serializable()
class CustomEncoder(keras.Model):
    def __init__(self, vocab_size, hidden_size, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size  # REQUIRED attribute
        
        # Your architecture here
        self.embedding = keras.layers.Embedding(vocab_size, hidden_size)
        self.transformer = ...  # Your layers
    
    def call(self, inputs, training=None):
        # inputs can be dict or tensor
        if isinstance(inputs, dict):
            input_ids = inputs["input_ids"]
        else:
            input_ids = inputs
        
        # Process
        hidden_states = self.embedding(input_ids)
        hidden_states = self.transformer(hidden_states, training=training)
        
        # REQUIRED: Return dict with "last_hidden_state"
        return {
            "last_hidden_state": hidden_states,
            "attention_mask": inputs.get("attention_mask")
        }

# Use with MLM
custom_encoder = CustomEncoder(vocab_size=30522, hidden_size=768)
mlm_model = MaskedLanguageModel(
    encoder=custom_encoder,
    vocab_size=30522,
    mask_token_id=103
)
```

### Continual Pre-training

Resume pre-training from a checkpoint:

```python
# Load previous checkpoint
mlm_model = keras.models.load_model("checkpoints/mlm_model_03.keras")

# Continue training with potentially different settings
mlm_model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-5)  # Lower LR
)

# Continue training
mlm_model.fit(
    train_dataset,
    epochs=5,
    initial_epoch=3  # Start from epoch 4
)
```

### Domain Adaptation

Adapt a pretrained model to a specific domain:

```python
# Load pretrained model
mlm_model = keras.models.load_model("pretrained_general.keras")

# Adjust masking for domain adaptation
mlm_model.mask_ratio = 0.10  # Gentler masking
mlm_model.mlm_head_dropout = 0.05  # Less dropout

# Compile with lower learning rate
mlm_model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-5, weight_decay=0.001)
)

# Train on domain-specific data
mlm_model.fit(medical_text_dataset, epochs=2)
```

4. **Data Pipeline**: Optimize `tf.data` with `.cache()` and `.prefetch()`

---

## Troubleshooting

### Common Issues

#### Loss is NaN or Exploding

**Problem**: Loss becomes NaN or explodes during training.

**Solutions**:
- Lower the learning rate (try 1e-5 or 2e-5)
- Check for corrupted data or encoding issues
- Increase gradient clipping: `gradient_clipping_by_norm=0.5`
- Ensure `layer_norm_eps` is not too small (use 1e-12)
- Enable mixed precision with dynamic loss scaling

```python
# Fix example
mlm_model.compile(
    optimizer=keras.optimizers.AdamW(
        learning_rate=1e-5,  # Lower LR
        gradient_clipnorm=0.5  # Clip gradients
    )
)
```

#### MLM Accuracy Not Improving

**Problem**: Accuracy stays flat or improves very slowly.

**Solutions**:
- Train longer (MLM requires many epochs on large data)
- Increase dataset size and diversity
- Check masking ratio (15% is optimal for most cases)
- Verify `special_token_ids` are correct
- Ensure `vocab_size` matches tokenizer exactly

```python
# Verify configuration
print(f"Vocab size: {tokenizer.vocab_size}")
print(f"Special tokens: {tokenizer.all_special_ids}")
print(f"Mask token: {tokenizer.mask_token_id}")
```

#### Out of Memory (OOM)

**Problem**: Training crashes with OOM error.

**Solutions**:
- Reduce batch size
- Reduce sequence length (`max_length=64` or `max_length=128`)
- Enable mixed precision
- Use gradient accumulation
- Consider smaller model variant

```python
# Memory-efficient configuration
keras.mixed_precision.set_global_policy('mixed_float16')

dataset = create_mlm_dataset(
    text_file,
    tokenizer,
    batch_size=16,  # Smaller batch
    max_length=64   # Shorter sequences
)
```

#### Slow Training

**Problem**: Training is slower than expected.

**Solutions**:
- Optimize data pipeline with `.cache()` and `.prefetch()`
- Use multi-GPU training
- Enable mixed precision
- Reduce logging frequency
- Pre-tokenize dataset offline

```python
# Optimized dataset
dataset = dataset.cache()  # Cache in memory
dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Prefetch batches
```

#### Serialization Errors

**Problem**: Cannot save or load model.

**Solutions**:
- Ensure encoder has `@keras.saving.register_keras_serializable()` decorator
- Verify `get_config()` includes all constructor parameters
- Use `.keras` format (not `.h5`)

```python
# Proper serialization
mlm_model.save("model.keras")  # Use .keras format
loaded = keras.models.load_model("model.keras")
```

---

## API Reference

### MaskedLanguageModel

```python
class MaskedLanguageModel(keras.Model):
    """Masked Language Model pre-trainer.
    
    Args:
        encoder: Encoder model with 'hidden_size' attribute
        vocab_size: Vocabulary size
        mask_ratio: Fraction of tokens to mask (default: 0.15)
        mask_token_id: ID for [MASK] token
        random_token_ratio: Fraction to replace with random (default: 0.1)
        unchanged_ratio: Fraction to leave unchanged (default: 0.1)
        special_token_ids: Token IDs to never mask
        mlm_head_activation: Activation for MLM head (default: "gelu")
        initializer_range: Weight initialization stddev (default: 0.02)
        mlm_head_dropout: Dropout rate for MLM head (default: 0.1)
        layer_norm_eps: LayerNorm epsilon (default: 1e-12)
    
    Attributes:
        encoder: The encoder being pretrained
        mlm_dense: Dense transformation layer
        mlm_dropout: Dropout layer
        mlm_norm: Layer normalization
        mlm_output: Output projection to vocabulary
    """
```

#### Methods

**`call(inputs, training=False)`**

Forward pass without masking (for inference).

- **inputs**: Dict with 'input_ids' or tensor
- **training**: Whether in training mode
- **Returns**: Logits (batch_size, seq_len, vocab_size)

**`train_step(data)`**

Custom training step with dynamic masking.

- **data**: Batch from dataset
- **Returns**: Dict of metrics

**`_mask_tokens(inputs)`**

Apply BERT-style masking to inputs.

- **inputs**: Dict with 'input_ids', 'attention_mask'
- **Returns**: (masked_inputs, labels, mask)

**`compute_loss(y, y_pred, sample_weight)`**

Compute MLM loss on masked positions.

- **y**: Original token IDs
- **y_pred**: Predicted logits
- **sample_weight**: Boolean mask
- **Returns**: Loss scalar

### Utility Functions

**`visualize_mlm_predictions(mlm_model, inputs, tokenizer, num_samples=4)`**

Visualize model predictions on masked tokens.

**`create_mlm_training_model(encoder, vocab_size, mask_token_id, ...)`**

Factory function to create compiled MLM model.

---

## References

1. **BERT Paper**: Devlin et al. (2018) - "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" - https://arxiv.org/abs/1810.04805

2. **RoBERTa**: Liu et al. (2019) - Improvements to BERT pre-training

3. **ELECTRA**: Clark et al. (2020) - Alternative pre-training approach

4. **Keras Documentation**: https://keras.io/

5. **dl_techniques Framework**: Internal documentation

--