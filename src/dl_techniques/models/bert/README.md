# BERT: Bidirectional Encoder Representations from Transformers

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A production-ready, fully-featured implementation of the **BERT (Bidirectional Encoder Representations from Transformers)** architecture in **Keras 3**. This implementation is based on the original paper by Devlin et al., providing a pure foundation model that separates the core encoding logic from task-specific heads for maximum flexibility.

This implementation follows the `dl_techniques` framework standards and modern Keras 3 best practices. It provides a modular, well-documented, and fully serializable model that works seamlessly across TensorFlow, PyTorch, and JAX backends. The architecture's key feature is its stack of **Transformer Encoder Layers**, designed to produce rich, contextualized token embeddings. It includes built-in support for loading standard model variants (`base`, `large`, etc.) with or without pretrained weights.

---

## Table of Contents

1. [Overview: What is BERT and Why It Matters](#1-overview-what-is-bert-and-why-it-matters)
2. [The Problem BERT Solves](#2-the-problem-bert-solves)
3. [How BERT Works: Core Concepts](#3-how-bert-works-core-concepts)
4. [Architecture Deep Dive](#4-architecture-deep-dive)
5. [Quick Start Guide](#5-quick-start-guide)
6. [Component Reference](#6-component-reference)
7. [Configuration & Model Variants](#7-configuration--model-variants)
8. [Comprehensive Usage Examples](#8-comprehensive-usage-examples)
9. [Advanced Usage Patterns](#9-advanced-usage-patterns)
10. [Performance Optimization](#10-performance-optimization)
11. [Training and Best Practices](#11-training-and-best-practices)
12. [Serialization & Deployment](#12-serialization--deployment)
13. [Testing & Validation](#13-testing--validation)
14. [Troubleshooting & FAQs](#14-troubleshooting--faqs)
15. [Technical Details](#15-technical-details)
16. [Citation](#16-citation)

---

## 1. Overview: What is BERT and Why It Matters

### What is BERT?

**BERT** is a landmark language representation model that uses the Transformer architecture to pre-train deep bidirectional representations from unlabeled text. Unlike previous models that were either context-free or unidirectional, BERT learns to understand the context of a word based on all of its surroundings (both left and right).

This implementation provides the core BERT encoder as a **foundation model**. It takes tokenized text as input and outputs a sequence of high-dimensional vectors, where each vector is a rich, contextualized embedding for a corresponding input token. These embeddings can then be used to fine-tune a wide variety of downstream NLP tasks.

### Key Innovations of this Implementation

1.  **Foundation Model Design**: The `BERT` class is a pure encoder. It is intentionally decoupled from any specific task, making it a reusable and flexible component for pre-training and multi-task learning.
2.  **Pretrained Weights Support**: The `from_variant` class method allows for easy instantiation of standard BERT configurations (`base`, `large`, `small`, `tiny`) and can automatically download and load official pretrained weights.
3.  **Keras 3 Native & Serializable**: Built as a composite `keras.Model`, the entire architecture, including all sub-layers, is fully serializable to the modern `.keras` format and compatible with TensorFlow, PyTorch, and JAX.
4.  **High Configurability**: The underlying `TransformerLayer` is highly configurable, allowing for experimentation with different normalization types, positions, FFN variants, and attention mechanisms.

### Why BERT Matters

**Traditional NLP Models**:
```
Model: Word2Vec / GloVe (Context-free)
  1. Generates a single, static vector representation for each word.
  2. Cannot distinguish between different meanings of a word in different
     contexts (e.g., "bank" of a river vs. financial "bank").

Model: Standard RNN/LSTM (Unidirectional)
  1. Processes text sequentially, building a representation of a word based
     only on the context that came before it.
  2. Fails to incorporate future context and struggles with long-range dependencies.

Model: ELMo (Shallowly Bidirectional)
  1. Concatenates independently trained left-to-right and right-to-left
     LSTMs. Not deeply bidirectional.
```

**BERT's Solution**:
```
BERT (Deeply Bidirectional Transformer):
  1. Uses a Transformer encoder architecture, where self-attention allows
     every token to attend to every other token in the sequence simultaneously.
  2. Its unique "Masked Language Model" (MLM) pre-training task enables it to
     learn deep bidirectional context.
  3. Result: Produces truly contextual embeddings that capture nuanced meaning,
     leading to state-of-the-art performance on a wide range of NLP tasks.
```

### Real-World Impact

This powerful contextual understanding has revolutionized NLP applications:

-   **Text Classification**: Sentiment analysis, topic categorization, and spam detection.
-   **Question Answering**: Systems like SQuAD that can find the exact answer to a question within a document.
-   **Named Entity Recognition (NER)**: Identifying people, organizations, and locations in text.
-   **Summarization & Paraphrasing**: Understanding text deeply enough to generate abstractive summaries.

---

## 2. The Problem BERT Solves

### The Challenge of Context in Language

Language is inherently ambiguous. The meaning of a word is determined by the words that surround it.

```
┌─────────────────────────────────────────────────────────────┐
│  The Challenge of Ambiguity and Context                     │
│                                                             │
│  1. Unidirectional Context (Left-to-Right):                 │
│     - "He went to the bank to..."                           │
│     - Based on this, a unidirectional model might guess     │
│       "withdraw money".                                     │
│                                                             │
│  2. Full Context (Bidirectional):                           │
│     - "He went to the bank to sit by the river."            │
│     - With the full sentence, we know "bank" refers to a    │
│       river bank. Unidirectional models would struggle to   │
│       correct their initial assumption.                     │
└─────────────────────────────────────────────────────────────┘
```

Relying on only past context is limiting. To truly understand language, a model needs to consider the entire sentence at once.

### How BERT's Bidirectional Architecture Changes the Game

BERT's architecture and pre-training method are designed specifically to overcome this limitation.

```
┌─────────────────────────────────────────────────────────────┐
│  The BERT Bidirectional Solution                            │
│                                                             │
│  1. Transformer Encoder: The self-attention mechanism is    │
│     not sequential. It looks at all words in the input at   │
│     the same time, allowing information to flow in both     │
│     directions from the very first layer.                   │
│                                                             │
│  2. Masked Language Model (MLM): To train this architecture,│
│     BERT randomly masks some words in a sentence and then   │
│     tries to predict them. To do this successfully, it MUST │
│     use both the left and right context surrounding the     │
│     masked word. This forces the model to learn deep        │
│     bidirectional relationships.                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. How BERT Works: Core Concepts

### The High-Level Architecture

The model transforms a sequence of token IDs into a sequence of rich, contextual vector representations.

```
┌──────────────────────────────────────────────────────────────────┐
│                     BERT Foundation Model Architecture           │
│                                                                  │
│ Input (Token IDs) ───►┌────────────────┐                         │
│                       │ BertEmbeddings │ (Token + Pos + Segment) │
│                       └────────┬───────┘                         │
│                                │                                 │
│                       ┌────────▼───────────┐                     │
│                       │ TransformerLayer   │ (Repeated N times)  │
│                       │(Self-Attention,FFN)│                     │
│                       └────────┬───────────┘                     │
│                                │                                 │
│                       ┌────────▼───────────┐                     │
│                       │Output Hidden States│ (Contextual Embeds) │
│                       └────────┬───────────┘                     │
│                                │                                 │
│                       ┌────────▼───────────┐                     │
│                       │ Task-Specific Head │ (e.g., Classifier)  │
│                       └────────────────────┘                     │
└──────────────────────────────────────────────────────────────────┘
```

### The Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     BERT Complete Data Flow                             │
└─────────────────────────────────────────────────────────────────────────┘

STEP 1: INPUT PREPARATION & EMBEDDING
─────────────────────────────────
Input Text -> Tokenizer -> Input Representation
    │
    ├─► Input IDs: Numerical IDs for each token (e.g., from WordPiece).
    ├─► Attention Mask: A binary mask (1 for real tokens, 0 for padding).
    ├─► Token Type IDs: Segment IDs to distinguish between sentences (e.g., for Q&A).
    │
    └─► BertEmbeddings Layer
        ├─► Token Embeddings
        ├─► Position Embeddings
        ├─► Token Type (Segment) Embeddings
        │
        └─► Summed Embeddings -> LayerNorm -> Dropout -> (B, seq_len, D)


STEP 2: BIDIRECTIONAL ENCODING
─────────────────────────────
Embedded Sequence (B, seq_len, D)
    │
    ├─► TransformerLayer 1
    │   ├─► [Norm] -> Multi-Head Self-Attention -> [Add & Norm]
    │   └─► [Norm] -> Feed-Forward Network      -> [Add & Norm]
    │
    ├─► TransformerLayer 2 ... N
    │
    └─► Final Hidden States (B, seq_len, D)


STEP 3: PROJECTION & FINE-TUNING
────────────────────────────────
Final Hidden States (B, seq_len, D)
    │
    ├─► For Sentence Classification:
    │   └─► Take the vector for the [CLS] token -> Dense -> Softmax
    │
    ├─► For Token Classification (e.g., NER):
    │   └─► Pass the entire sequence of vectors -> Dense -> Softmax per token
    │
    └─► Output for a specific downstream task.
```

---

## 4. Architecture Deep Dive

### 4.1 `BertEmbeddings` Layer

A custom layer responsible for creating the initial input representation.
-   It takes `input_ids`, `token_type_ids`, and `position_ids` as input.
-   It contains three separate embedding tables: for words, positions, and sentence segments.
-   The final embedding is the element-wise sum of these three components, followed by Layer Normalization and Dropout. This combined representation gives the model a strong initial signal about the token's identity, its position, and which sentence it belongs to.

### 4.2 `TransformerLayer`

The core building block of the BERT encoder, repeated multiple times.
-   This implementation provides a highly configurable `TransformerLayer` that encapsulates a standard encoder block.
-   **Multi-Head Self-Attention**: Allows the model to weigh the importance of all other words in the sequence when encoding a specific word, capturing complex dependencies.
-   **Feed-Forward Network (FFN)**: A two-layer MLP applied to each position independently, providing non-linear transformation capacity.
-   **Residual Connections & Normalization**: Each sub-layer (attention and FFN) is wrapped with a residual connection and layer normalization, which is crucial for training deep networks. This implementation supports both Pre-Norm and Post-Norm configurations.

### 4.3 Task-Specific Heads

This implementation deliberately separates the core BERT model from task heads. The `dl_techniques.nlp.heads` module and the `create_bert_with_head` factory function are the intended integration points.
-   The heads are simple Keras layers that take the output from `BERT` (`last_hidden_state`) and project it to the desired output for a specific task.
-   Examples: A `Dense` layer for classification, a `TokenClassificationHead` for NER, etc.

---

## 5. Quick Start Guide

### Installation

```bash
# Ensure you have the required dependencies
pip install keras>=3.0 tensorflow>=2.16 numpy
```

### Your First BERT Model (30 seconds)

Let's build a BERT-based sentiment analysis model.

```python
import keras
import numpy as np

from dl_techniques.models.bert import create_bert_with_head
from dl_techniques.layers.nlp_heads import NLPTaskConfig, NLPTaskType

# 1. Define the downstream task
sentiment_config = NLPTaskConfig(
    name="sentiment_analysis",
    task_type=NLPTaskType.SENTIMENT_ANALYSIS,
    num_classes=3  # e.g., positive, negative, neutral
)

# 2. Create a BERT model with the sentiment head
# This factory function will:
#  - Instantiate a BERT-base encoder
#  - Instantiate a classification head
#  - Combine them into a single Keras model
print("🚀 Creating BERT-base model with a sentiment analysis head...")
model = create_bert_with_head(
    bert_variant="base",
    task_config=sentiment_config,
    pretrained=False  # Set to True to download pretrained weights
)

# 3. Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=2e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
print("✅ BERT model created and compiled successfully!")
model.summary()

# 4. Use the model with dummy data
# In a real application, you would use a BERT tokenizer here.
BATCH_SIZE, SEQ_LEN = 4, 128
dummy_inputs = {
    "input_ids": np.random.randint(0, 30522, size=(BATCH_SIZE, SEQ_LEN)),
    "attention_mask": np.ones((BATCH_SIZE, SEQ_LEN), dtype="int32"),
    "token_type_ids": np.zeros((BATCH_SIZE, SEQ_LEN), dtype="int32")
}
dummy_labels = np.random.randint(0, 3, size=(BATCH_SIZE,))

# The model is ready for training
# model.fit(dummy_inputs, dummy_labels, epochs=1)

# Make a prediction
predictions = model.predict(dummy_inputs)
print(f"\nOutput predictions shape: {predictions.shape}") # (4, 3)
```

---

## 6. Component Reference

### 6.1 `BERT` (Model Class)

**Purpose**: The main Keras `Model` subclass that implements the core BERT encoder. It outputs a dictionary containing `last_hidden_state`.

```python
from dl_techniques.models.bert  import BERT

# Create a standalone BERT-base encoder
# Load pretrained weights from default URL
bert_encoder = BERT.from_variant("base", pretrained=True)

# Create a custom BERT model
custom_encoder = BERT.from_variant("small", vocab_size=50000)
```

### 6.2 Factory Functions

#### `create_bert_with_head(...)`
The recommended high-level factory for creating a complete, end-to-end model for a specific NLP task. It handles the creation and connection of both the BERT encoder and the task head.

---

## 7. Configuration & Model Variants

This implementation provides several standard configurations based on the original paper.

| Variant | Hidden Size | Layers | Heads | Parameters | Use Case |
|:---:|:---:|:---:|:---:|:---:|:---|
| **`tiny`** | 256 | 4 | 4 | ~15M | Quick tests, mobile/edge deployment |
| **`small`**| 512 | 6 | 8 | ~40M | Resource-constrained environments |
| **`base`** | 768 | 12 | 12 | ~110M | Standard for most applications |
| **`large`** | 1024| 24 | 16 | ~340M | Maximum performance, requires significant compute |

### Customizing the Configuration

You can override default parameters of a variant by passing them as keyword arguments to `from_variant`.

```python
from dl_techniques.models.bert import BERT
# Create a BERT-base model but with a larger vocab size
# This is useful when fine-tuning on a domain with new vocabulary
model = BERT.from_variant(
    "base",
    pretrained=True, # Will load weights, skipping the embedding layer
    vocab_size=50000
)
```

---

## 8. Comprehensive Usage Examples

### Example 1: Loading Pretrained Weights

The `pretrained` argument is flexible:
```python
from dl_techniques.models.bert import BERT

# 1. Download and load default 'uncased' weights for BERT-base
model = BERT.from_variant("base", pretrained=True)

# 2. Download and load 'cased' weights for BERT-large
model = BERT.from_variant("large", pretrained=True, weights_dataset="cased")

# 3. Load weights from a local file path
model = BERT.from_variant("base", pretrained="/path/to/my/bert_weights.keras")
```

### Example 2: Building a Model for Named Entity Recognition (NER)

NER is a token-level classification task. The factory handles this seamlessly.

```python
from dl_techniques.models.bert import create_bert_with_head
from dl_techniques.layers.nlp_heads.task_types import NLPTaskConfig, NLPTaskType

# 1. Define the NER task configuration
ner_config = NLPTaskConfig(
    name="ner",
    task_type=NLPTaskType.NAMED_ENTITY_RECOGNITION,
    num_classes=9  # e.g., O, B-PER, I-PER, B-LOC, I-LOC, etc.
)

# 2. Create the full model with a pretrained BERT encoder
ner_model = create_bert_with_head(
    bert_variant="base",
    task_config=ner_config,
    pretrained=True
)

# 3. Inspect the model
ner_model.summary()

# The output shape will be (batch_size, sequence_length, num_classes)
# perfect for training on a token-level objective.
```

---

## 9. Advanced Usage Patterns

### Pattern 1: Using BERT as a Feature Extractor

You can use the raw output of the `BERT` encoder as rich contextual features for other models.

```python
import keras
from dl_techniques.models.bert import BERT
from dl_techniques.layers.nlp_heads.task_types import NLPTaskConfig, NLPTaskType

# 1. Create the standalone BERT encoder
bert_encoder = BERT.from_variant("base", pretrained=True)

# 2. Define your own downstream model that uses BERT's outputs
inputs = {
    "input_ids": keras.Input(shape=(None,), dtype="int32"),
    "attention_mask": keras.Input(shape=(None,), dtype="int32"),
    "token_type_ids": keras.Input(shape=(None,), dtype="int32")
}
bert_outputs = bert_encoder(inputs)
sequence_output = bert_outputs["last_hidden_state"] # Shape: (batch, seq_len, 768)

# Example: Add a BiLSTM on top of BERT features for NER
downstream_layer = keras.layers.Bidirectional(
    keras.layers.LSTM(128, return_sequences=True)
)(sequence_output)
outputs = keras.layers.Dense(9, activation="softmax")(downstream_layer)

feature_model = keras.Model(inputs, outputs)
feature_model.summary()
```

### Pattern 2: Multi-Task Fine-Tuning

Share a single BERT encoder across multiple tasks to improve performance and efficiency.

```python
import keras
from dl_techniques.models.bert import BERT
from dl_techniques.layers.nlp_heads import NLPTaskConfig, NLPTaskType, TextClassificationHead, TokenClassificationHead

# 1. Create one shared BERT encoder
bert_encoder = BERT.from_variant("base", pretrained=True)
bert_encoder.trainable = True # Fine-tune the encoder

# 2. Define inputs
inputs = { ... } # Same as above

# 3. Get shared features
bert_outputs = bert_encoder(inputs)

# 4. Create two different heads
sentiment_head = TextClassificationHead(num_classes=2, name="sentiment")
ner_head = TokenClassificationHead(num_classes=9, name="ner")

# 5. Get task-specific outputs
sentiment_output = sentiment_head(bert_outputs)
ner_output = ner_head(bert_outputs)

# 6. Build the multi-output model
multi_task_model = keras.Model(
    inputs=inputs,
    outputs={"sentiment": sentiment_output, "ner": ner_output}
)
```

---

## 10. Performance Optimization

### Mixed Precision Training

For `base` and `large` models, mixed precision can dramatically speed up training and reduce memory usage on compatible GPUs.

```python
import keras
from dl_techniques.models.bert import BERT

# Enable mixed precision globally
keras.mixed_precision.set_global_policy('mixed_float16')

# Create model (will automatically use mixed precision)
model = BERT.from_variant("large", pretrained=True)
# ... compile and fit ...
```

### XLA Compilation

Use `jit_compile=True` for graph compilation, which can provide a significant speed boost after an initial compilation warmup.

```python
from dl_techniques.models.bert import create_bert_with_head
model = create_bert_with_head(...)
model.compile(optimizer="adam", loss="...", jit_compile=True)
# Keras will now attempt to compile the training step with XLA.
```

---

## 11. Training and Best Practices

### Fine-Tuning Strategy

A common best practice is to use a differential learning rate: a small learning rate for the pretrained BERT layers (e.g., 2e-5 to 5e-5) and a larger learning rate for the newly initialized task head. This prevents catastrophic forgetting in the pretrained body while allowing the new head to adapt quickly.

### Handling Long Sequences

The self-attention mechanism in BERT has a memory and computational complexity of O(n²), where n is the sequence length. This makes it challenging to use with very long documents. Standard practice is to truncate text to BERT's maximum length (typically 512 tokens). For longer documents, strategies like sliding windows or using specialized models (e.g., Longformer) are required.

### Input Representation

When fine-tuning a pretrained BERT model, it is **critical** to use the exact same tokenizer and input format that was used during pre-training. This includes the specific vocabulary, subword tokenization rules (e.g., WordPiece), and the use of special tokens like `[CLS]` and `[SEP]`.

---

## 12. Serialization & Deployment

The `BERT` model and all its custom layers are fully serializable using Keras 3's modern `.keras` format, thanks to the `@keras.saving.register_keras_serializable()` decorator.

### Saving and Loading

```python
import keras
from dl_techniques.models.bert import create_bert_with_head
# Create and train a model
model = create_bert_with_head(...)
# model.compile(...) and model.fit(...)

# Save the entire model
model.save('my_bert_ner_model.keras')

# Load the model in a new session without needing custom_objects
loaded_model = keras.models.load_model('my_bert_ner_model.keras')
```

---

## 13. Testing & Validation

### Unit Tests

```python
import keras
import numpy as np
from dl_techniques.models.bert import create_bert_with_head, BERT

def test_model_creation_all_variants():
    """Test model creation for all variants."""
    for variant in BERT.MODEL_VARIANTS.keys():
        model = BERT.from_variant(variant)
        assert model is not None
        print(f"✓ BERT-{variant} created successfully")

def test_forward_pass_shape():
    """Test the output shape of a forward pass."""
    model = BERT.from_variant("tiny")
    dummy_input = {
        "input_ids": np.random.randint(0, 30522, (4, 64)),
        "attention_mask": np.ones((4, 64), dtype="int32")
    }
    output = model(dummy_input)
    hidden_state = output["last_hidden_state"]
    # (batch, seq_len, hidden_size)
    assert hidden_state.shape == (4, 64, model.hidden_size)
    print("✓ Forward pass has correct shape")

# Run tests
if __name__ == '__main__':
    test_model_creation_all_variants()
    test_forward_pass_shape()
    print("\n✅ All tests passed!")
```

---

## 14. Troubleshooting & FAQs

**Issue 1: Out-of-Memory (OOM) errors during training.**

-   **Cause**: BERT is very memory-intensive due to the self-attention mechanism's quadratic complexity. `BERT-large` and long sequences (512) with a large batch size can easily exceed GPU memory.
-   **Solution**: 1) Reduce the `batch_size`. 2) Decrease the sequence length by truncating inputs more aggressively. 3) Use gradient accumulation to simulate a larger batch size. 4) Switch to a smaller model variant (e.g., `base` instead of `large`). 5) Enable mixed precision training.

**Issue 2: Mismatched shapes when loading pretrained weights.**

-   **Cause**: You have customized the architecture (e.g., `vocab_size`) in a way that is incompatible with the saved weights' shapes.
-   **Solution**: This implementation's `load_pretrained_weights` method automatically uses `skip_mismatch=True`. It will log a warning that it is skipping incompatible layers (like the embedding layer if `vocab_size` is changed), which is often the desired behavior.

### Frequently Asked Questions

**Q: Why is BERT a "foundation model"?**

A: The term "foundation model" refers to large models pre-trained on vast amounts of data that can be adapted (or "fine-tuned") to a wide range of downstream tasks. BERT is a prime example because its pre-trained language understanding capabilities serve as a powerful base for almost any NLP task, from classification to question answering. This implementation honors that concept by keeping the core encoder separate and reusable.

**Q: What is the `[CLS]` token for?**

A: The `[CLS]` token is a special token added to the beginning of every input sequence. Because BERT's self-attention mechanism allows every token to interact with every other, the final hidden state corresponding to the `[CLS]` token can be thought of as an aggregate representation of the entire sequence. It is conventionally used as the input to a classifier for sentence-level tasks like sentiment analysis.

---

## 15. Technical Details

### The Power of Bidirectional Self-Attention

The core of BERT is the `MultiHeadSelfAttention` mechanism inside each `TransformerLayer`. Unlike an RNN which processes one token at a time, self-attention computes the representation of each token by attending to all other tokens in the input sequence. This allows the model to draw connections between distant words and build a holistic understanding of the sentence from the very first layer.

### Masked Language Model (MLM)

BERT's bidirectionality is made possible by its pre-training objective, the Masked Language Model. During pre-training, the model is fed sentences where ~15% of the tokens are randomly masked (replaced with a `[MASK]` token). The model's task is to predict the original identity of these masked tokens. To do this, it is forced to rely on the surrounding unmasked tokens from both the left and the right, thereby learning a deep, bidirectional representation of language.

### Pre-Norm vs. Post-Norm

The original BERT paper used "Post-Norm" architecture (Sublayer -> Add -> Norm). However, subsequent research found that "Pre-Norm" (Norm -> Sublayer -> Add) leads to more stable training, especially for very deep models. This implementation's `TransformerLayer` is configurable via the `normalization_position` argument, allowing you to use either, though the `BERT` class defaults to the original Post-Norm for compatibility.

---

## 16. Citation

This implementation is based on the original BERT paper. If you use this model or its concepts in your research, please cite the foundational work:

```bibtex
@inproceedings{devlin2019bert,
  title={{BERT}: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  booktitle={Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)},
  pages={4171--4186},
  year={2019},
  publisher={Association for Computational Linguistics}
}
```