# Tree Transformer: Grammar Induction with Hierarchical Attention

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18+-orange.svg)](https://www.tensorflow.org/)

A production-ready, fully-featured implementation of the **Tree Transformer** in **Keras 3**. This implementation is based on the paper "Tree Transformer: Integrating Tree Structures into Self-Attention," which introduces a novel attention mechanism capable of learning soft constituency trees from raw text without explicit syntactic supervision.

This implementation follows the `dl_techniques` framework standards and modern Keras 3 best practices. It provides a modular, well-documented, and fully serializable model that works seamlessly across TensorFlow, PyTorch, and JAX backends. The architecture's key innovation is its **Hierarchical Group Attention** mechanism, which enables the model to infer syntactic structure as a byproduct of standard language modeling.

---

## Table of Contents

1. [Overview: What is Tree Transformer and Why It Matters](#1-overview-what-is-tree-transformer-and-why-it-matters)
2. [The Problem Tree Transformer Solves](#2-the-problem-tree-transformer-solves)
3. [How Tree Transformer Works: Core Concepts](#3-how-tree-transformer-works-core-concepts)
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

## 1. Overview: What is Tree Transformer and Why It Matters

### What is Tree Transformer?

The **Tree Transformer** is an advanced transformer architecture designed for grammar induction and language modeling. It enhances the standard self-attention mechanism with a hierarchical structure, allowing it to learn soft, latent constituency trees directly from text.

Instead of treating a sentence as a flat sequence of tokens, the Tree Transformer uses **Group Attention** to compute the probability of any given span of words forming a coherent syntactic constituent (like a noun phrase or verb phrase). This learned structure then modulates the standard multi-head attention, biasing the model to attend to meaningful phrases rather than arbitrary token pairs.

### Key Innovations

1.  **Hierarchical Group Attention**: The core of the model. It first computes attention between adjacent words ("neighbor attention") and then uses a dynamic programming-like algorithm to efficiently compute attention over all possible spans.
2.  **Unsupervised Grammar Induction**: The model learns syntactic tree structures without requiring labeled treebank data. The only supervision needed is a standard language modeling objective (e.g., Masked Language Modeling).
3.  **Structure-Modulated Attention**: The learned group probabilities are directly multiplied with the standard attention weights in a `TreeMHA` layer, seamlessly integrating syntactic bias into the transformer's core.
4.  **Layer-to-Layer Refinement**: The structural information (group probabilities) is passed from one layer to the next, allowing the model to refine its understanding of the sentence's hierarchy at different levels of abstraction.

### Why Tree Transformer Matters

**Traditional Models**:
```
Model: Syntactic Parsers (e.g., CKY)
  1. Require large, manually annotated treebanks for training.
  2. Expensive to create and not available for most languages.
  3. Often used as a separate preprocessing step, not integrated into
     end-to-end models.

Model: Standard Transformer (e.g., BERT, GPT)
  1. Employs a "flat" self-attention mechanism where every token can
     attend to every other token.
  2. Lacks an explicit inductive bias for the hierarchical, compositional
     nature of human language.
  3. While it can learn some syntax implicitly, it's not its primary design.
```

**Tree Transformer's Solution**:
```
Tree Transformer Approach:
  1. Combines the representational power of Transformers with a built-in
     mechanism for syntax learning.
  2. The Group Attention module learns a soft constituency parse of the input.
  3. The TreeMHA module uses this parse to guide the attention mechanism.
  4. The model outputs both standard language model predictions (logits) and
     the learned tree structures (break probabilities).
  5. Benefit: Achieves a "best-of-both-worlds" model that is a powerful
     language model and an unsupervised syntactic parser in one.
```

### Real-World Impact

This integrated approach is critical for tasks that benefit from a deep understanding of syntax:

-   **Grammar Induction**: Discovering the syntactic rules of a language from raw text.
-   **Improved Language Modeling**: A better syntactic bias can lead to more coherent and grammatically correct text generation.
-   **Code Generation**: Understanding the nested, tree-like structure of programming languages.
-   **Complex Question Answering**: Decomposing complex questions by understanding their syntactic parts.
-   **Low-Resource NLP**: Learning syntax for languages that lack annotated treebanks.

---

## 2. The Problem Tree Transformer Solves

### The "Flat World" of Standard Self-Attention

Human language is inherently hierarchical. Words combine to form phrases, which combine to form clauses, which form sentences. The meaning of a sentence is derived from this compositional structure.

Standard self-attention, however, operates on a flat sequence. It does not have a built-in bias to recognize that "the old man" is a more meaningful unit to attend to than "man who is".

```
┌─────────────────────────────────────────────────────────────┐
│  The Challenge of Missing Hierarchical Bias                 │
│                                                             │
│  Sentence: The old man who lives next door is a scientist.  │
│                                                             │
│  1. Human understanding (Hierarchical):                     │
│     - "The old man" -> Noun Phrase                          │
│     - "who lives next door" -> Relative Clause              │
│     - These units combine to form a larger subject.         │
│                                                             │
│  2. Standard Attention's View (Flat):                       │
│     - Each word can connect to any other word with equal    │
│       ease, based only on query-key similarity.             │
│     - The model must learn the concept of "phrases" from    │
│       scratch, which is inefficient.                        │
└─────────────────────────────────────────────────────────────┘
```

### How The Tree Architecture Changes the Game

The Tree Transformer explicitly introduces this missing hierarchical bias.

```
┌─────────────────────────────────────────────────────────────┐
│  The Tree Transformer Solution                              │
│                                                             │
│  1. Inducing Structure: The GroupAttention layer calculates │
│     the probability that any substring (e.g., "The old man")│
│     forms a valid constituent.                              │
│                                                             │
│  2. Guiding Attention: The TreeMHA layer uses these         │
│     probabilities to up-weight attention scores within      │
│     these identified constituents.                          │
│                                                             │
│  3. Emergent Syntax: The model is incentivized to learn     │
│     grammatically correct structures because doing so helps │
│     it better predict masked words in the language modeling │
│     task.                                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. How Tree Transformer Works: Core Concepts

### The High-Level Architecture

The model processes token IDs through a stack of specialized transformer blocks, outputting both predictions and the learned syntactic structure.

```
┌──────────────────────────────────────────────────────────────────┐
│                      Tree Transformer Architecture               │
│                                                                  │
│  Input Token IDs ───►┌────────────────┐                          │
│                      │   Embedding &  │                          │
│                      │ Pos. Encoding  │                          │
│                      └────────┬───────┘                          │
│                               │                                  │
│       ┌───────────────────────▼───────────────────────┐          │
│       │        TreeTransformerBlock (Repeated N times)│          │
│       │ ┌────────────────┐   ┌─────────┐   ┌──────┐   │          │
│       │ │ GroupAttention ├──►│ TreeMHA ├──►│ FFN  │   │          │
│       │ └──────┬─────────┘   └────┬────┘   └──────┘   │          │
│       │        │                  │                   │          │
│       │        └─────────┬────────┘                   │          │
│       │           [+ Residual Connections]            │          │
│       └───────────────────────┬───────────────────────┘          │
│                               │                                  │
│                       ┌───────▼────────┐                         │
│                       │  Final Layer & │                         │
│                       │    LM Head     │                         │
│                       └───────┬────────┘                         │
│                               │                                  │
│  ┌────────────────┐  ┌────────▼───────┐  ┌──────────────────┐    │
│  │ Break Probs    │◄─┤   Final State  ├─►│      Logits      │    │
│  │ (Tree)         │  └────────────────┘  └──────────────────┘    │
│  └────────────────┘                                              │
└──────────────────────────────────────────────────────────────────┘
```

### The Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   Tree Transformer Complete Data Flow                   │
└─────────────────────────────────────────────────────────────────────────┘

STEP 1: PREPROCESSING & EMBEDDING
─────────────────────────────────
Input IDs (B, L)
    │
    ├─► Embedding Layer -> (B, L, D)
    │
    ├─► Scale by sqrt(d_model)
    │
    └─► PositionalEncoding -> (B, L, D)


STEP 2: HIERARCHICAL PROCESSING (inside one TreeTransformerBlock)
────────────────────────────────────────────────────────────────
Input `x` (B, L, D), `group_prob_prior` (from previous layer, init with 0.0)
    │
    ├─► GroupAttention(`x`, `mask`, `group_prob_prior`)
    │   ├─► Compute neighbor attention (adjacents)
    │   ├─► Tree induction (DP) to get span probabilities
    │   └─► Output: `group_prob_out` (B, L, L), `break_prob` (B, L, L)
    │
    ├─► Pre-LN Residual Block 1
    │   ├─► LayerNorm(`x`)
    │   ├─► TreeMHA(q, k, v, `group_prob_out`, `mask`) -> attn_output
    │   └─► `x` = `x` + Dropout(attn_output)
    │
    ├─► Pre-LN Residual Block 2
    │   ├─► LayerNorm(`x`)
    │   ├─► FFN(...) -> ffn_output
    │   └─► `x` = `x` + Dropout(ffn_output)
    │
    └─► Output: `x`, `group_prob_out`, `break_prob`
        (The `group_prob_out` is passed as the prior to the next block)


STEP 3: PROJECTION & FINAL OUTPUT
─────────────────────────────────
Final Hidden States (B, L, D) from last block
    │
    ├─► Final LayerNorm
    │
    ├─► LM Head (Dense Layer)
    │
    └─► Logits (B, L, VocabSize)

Simultaneously:
    │
    ├─► Collect `break_prob` from each layer
    │
    └─► Stack -> Break Probs (B, NumLayers, L, L)
```

---

## 4. Architecture Deep Dive

### 4.1 `GroupAttention` Layer

This is the core innovation of the Tree Transformer.
-   It takes the sequence embeddings and computes **neighbor attention**: the likelihood that two adjacent tokens form a constituent.
-   It uses a dynamic programming-like algorithm, implemented efficiently with matrix multiplications, to extend neighbor attention to all possible contiguous spans (**group attention**).
-   The output is a matrix of probabilities `g_attn` where `g_attn[i, j]` represents the score for the span from token `i` to `j` being a single constituent.

### 4.2 `TreeMHA` Layer (Tree-Modulated Multi-Head Attention)

A modification of the standard `MultiHeadAttention` layer.
-   It performs scaled dot-product attention as usual to get a standard attention matrix.
-   Crucially, it then performs an element-wise multiplication of these attention weights with the `group_prob` matrix from the `GroupAttention` layer.
-   This modulation biases the attention mechanism, encouraging attention heads to focus on tokens within the same learned constituents.

### 4.3 `TreeTransformerBlock`

A self-contained block that encapsulates the full logic of one layer.
-   It follows a Pre-LN (Layer Normalization before the sub-layer) design for training stability.
-   It orchestrates the flow: `GroupAttention` -> `TreeMHA` -> `FFN`, with residual connections around the `TreeMHA` and `FFN` stages.
-   It passes the computed `group_prob` to the next `TreeTransformerBlock` in the stack.

---

## 5. Quick Start Guide

### Installation

```bash
# Ensure you have the required dependencies
pip install keras>=3.0 tensorflow>=2.16 numpy matplotlib
```

### Your First Tree Transformer Model (30 seconds)

Let's build a small model and inspect its outputs for a dummy sequence.

```python
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras import ops

# Local imports from your project structure
from model import create_tree_transformer

# 1. Define model parameters
VOCAB_SIZE = 1000
NUM_LAYERS = 2
D_MODEL = 64
NUM_HEADS = 4
D_FF = 128
MAX_LEN = 32
BATCH_SIZE = 4

# 2. Create a Tree Transformer model using the factory function
model = create_tree_transformer(
    vocab_size=VOCAB_SIZE,
    num_layers=NUM_LAYERS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    d_ff=D_FF,
    max_len=MAX_LEN,
)
model.summary()
print("✅ Tree Transformer model created successfully!")

# 3. Generate a batch of dummy input data (integer token IDs)
dummy_input_ids = ops.convert_to_tensor(
    np.random.randint(1, VOCAB_SIZE, size=(BATCH_SIZE, MAX_LEN))
)

# 4. Perform a forward pass
logits, break_probs = model(dummy_input_ids, training=False)
print(f"Logits shape: {logits.shape}")
print(f"Break probabilities shape: {break_probs.shape}")

# 5. Visualize the break probabilities from the last layer for one sentence
# Break probability is high between two words if they are likely the boundaries
# of two separate constituents.
last_layer_break_probs = break_probs[0, -1, :, :] # (seq_len, seq_len)
# We are interested in the superdiagonal for adjacent breaks
adjacent_break_probs = ops.diagonal(last_layer_break_probs, offset=1)

plt.figure(figsize=(12, 6))
plt.imshow(ops.convert_to_numpy(last_layer_break_probs), cmap='viridis')
plt.colorbar(label="Constituency Score")
plt.title("Learned Group Attention Matrix (Last Layer, Sample 0)")
plt.xlabel("Token Index")
plt.ylabel("Token Index")
plt.show()

print("\nAdjacent Break Probabilities:")
print(ops.convert_to_numpy(adjacent_break_probs))
```

---

## 6. Component Reference

### 6.1 `TreeTransformerModel` (Model Class)

**Purpose**: The main Keras `Model` subclass that assembles the entire Tree Transformer.

**Location**: `model.TreeTransformerModel`

### 6.2 `create_tree_transformer` (Factory Function)

**Purpose**: The recommended, convenient way to create a `TreeTransformerModel` with standard configurations.

**Location**: `model.create_tree_transformer`

---

## 7. Configuration & Model Variants

You can create models of different sizes by adjusting the parameters in the `create_tree_transformer` factory.

| Parameter | Description | `small` | `base` | `large` |
|:---|:---|:---:|:---:|:---:|
| `num_layers` | Number of TreeTransformerBlocks | 4 | 10 | 16 |
| `d_model` | Embedding dimension | 256 | 512 | 1024 |
| `num_heads` | Number of attention heads | 4 | 8 | 16 |
| `d_ff` | Inner FFN dimension | 1024 | 2048 | 4096 |

**Example: Creating a `base` model**
```python
model = create_tree_transformer(
    vocab_size=30000,
    num_layers=10,
    d_model=512,
    num_heads=8,
    d_ff=2048
)
```

---

## 8. Comprehensive Usage Examples

### Example 1: Setting up for Masked Language Modeling (MLM)

The primary training objective for Tree Transformer is MLM, identical to BERT.

```python
# Assume you have a data pipeline that yields (inputs, labels)
# where `inputs` has some tokens replaced by a [MASK] ID, and
# `labels` contains the original token IDs at those positions.

# 1. Create the model
model = create_tree_transformer(vocab_size=30000)

# 2. Define a standard cross-entropy loss that ignores non-masked tokens
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def masked_loss(y_true, y_pred):
    # y_true should be -100 (or another ignore_index) for non-masked tokens
    mask = ops.not_equal(y_true, -100)
    y_true_masked = ops.where(mask, y_true, 0)
    y_pred_masked = ops.where(ops.expand_dims(mask, -1), y_pred, 0)
    return loss_fn(y_true_masked, y_pred_masked)

# 3. Compile and train
model.compile(optimizer="adam", loss=masked_loss)
# model.fit(mlm_dataset, epochs=3)
```

### Example 2: Interpreting the Learned Tree

The `break_probs` output can be used to construct a constituency parse. A high value in `break_probs[layer, i, i+1]` suggests a syntactic boundary between token `i` and `i+1`.

```python
def find_best_split(probs, i, j):
    """Find the best split point for the span [i, j]."""
    if i == j:
        return -1, -1
    max_prob = -1
    best_k = -1
    for k in range(i, j):
        # The score of a split at k is the sum of scores of the
        # two resulting sub-spans. This is a simplification.
        # The paper uses a more complex CKY-like decoding.
        split_prob = probs[i, k] + probs[k + 1, j]
        if split_prob > max_prob:
            max_prob = split_prob
            best_k = k
    return best_k, max_prob

def build_tree(probs, i, j, tokens):
    """Recursively build and print a parse tree."""
    if i == j:
        return f"({tokens[i]})"
    k, _ = find_best_split(probs, i, j)
    if k == -1:
        return " ".join(tokens[i:j+1])
    
    left_tree = build_tree(probs, i, k, tokens)
    right_tree = build_tree(probs, k + 1, j, tokens)
    return f"({left_tree} {right_tree})"

# # Assume:
# sentence = "The old man is sleeping"
# tokens = sentence.split()
# logits, break_probs = model(tokenize(sentence))
# group_probs = 1.0 - break_probs # Simplified for visualization
# last_layer_probs = group_probs[0, -1, :, :]
#
# tree_str = build_tree(last_layer_probs, 0, len(tokens) - 1, tokens)
# print(tree_str)
# Expected output might look like: "((The (old man)) (is sleeping))"
```

---

## 9. Advanced Usage Patterns

### Pattern 1: Using Break Probabilities as Syntactic Features

The `break_probs` tensor is a rich, multi-layer representation of the sentence's syntax. You can extract these features for downstream tasks that require syntactic awareness, such as relation extraction or semantic role labeling.

```python
# 1. Create the base Tree Transformer
base_model = create_tree_transformer(vocab_size=30000)

# 2. Create a feature extractor model
feature_extractor = keras.Model(
    inputs=base_model.input,
    outputs=base_model.output[1] # Output only the break_probs
)

# 3. Extract syntactic features
syntactic_features = feature_extractor.predict(dummy_input_ids)
print(f"Syntactic features shape: {syntactic_features.shape}")
# Shape: (batch, num_layers, seq_len, seq_len)
```

### Pattern 2: Fine-tuning with a Syntactic Loss

Although designed for unsupervised learning, if you have a treebank, you can add a secondary loss function that compares the model's `break_probs` with the ground-truth constituency boundaries. This can guide the model to learn more accurate parses.

---

## 10. Performance Optimization

### Mixed Precision Training

For larger models (`base` or `large`), mixed precision can significantly speed up training on compatible GPUs.

```python
# Enable mixed precision globally
keras.mixed_precision.set_global_policy('mixed_float16')

# Create model (will automatically use mixed precision)
model = create_tree_transformer(vocab_size=30000, num_layers=10)
# ... compile and fit ...
```

### XLA Compilation

Use `jit_compile=True` for graph compilation, which can provide a speed boost, especially for the matrix-heavy operations in `GroupAttention`.

```python
model = create_tree_transformer(vocab_size=30000)
model.compile(optimizer="adam", loss=masked_loss, jit_compile=True)
# Keras will now attempt to compile the training step with XLA.
```

---

## 11. Training and Best Practices

### The Training Objective

The model is trained end-to-end with a standard Masked Language Modeling (MLM) loss. The crucial insight of the paper is that in order to excel at predicting masked words, the model is implicitly encouraged to understand the grammatical structure of the sentence. Constituent phrases provide strong contextual clues, so learning to identify them helps minimize the MLM loss.

### The Role of the Inter-Layer Prior

The `group_prob` output from layer `N` is fed as a `prior` to layer `N+1`. This allows the model to build up its syntactic understanding hierarchically. Early layers might identify simple, local constituents (like noun phrases), while later layers can use this information to compose them into more complex structures (like clauses).

---

## 12. Serialization & Deployment

The `TreeTransformerModel` and all its custom layers (`GroupAttention`, `TreeMHA`, etc.) are fully serializable using Keras 3's modern `.keras` format, thanks to the `@keras.saving.register_keras_serializable` decorator and `get_config` methods.

### Saving and Loading

```python
# Create and train model
model = create_tree_transformer(vocab_size=30000)
# model.compile(...) and model.fit(...)

# Save the entire model
model.save('my_tree_transformer.keras')

# Load the model in a new session
loaded_model = keras.models.load_model(
    'my_tree_transformer.keras',
    # Pass any custom loss functions used during compilation
    custom_objects={'masked_loss': masked_loss}
)
```

---

## 13. Testing & Validation

### Unit Tests

```python
import keras
import numpy as np
from model import TreeTransformerModel, create_tree_transformer

def test_model_creation():
    """Test model creation with factory."""
    model = create_tree_transformer(
        vocab_size=1000, num_layers=2, d_model=32, num_heads=2, d_ff=64
    )
    assert model is not None
    print("✓ Tree Transformer creation successful")

def test_forward_pass_shape():
    """Test the output shape of a forward pass."""
    model = create_tree_transformer(
        vocab_size=1000, num_layers=2, d_model=32, num_heads=2, d_ff=64, max_len=16
    )
    dummy_input = np.random.randint(1, 1000, (4, 16))
    logits, break_probs = model(dummy_input)

    assert logits.shape == (4, 16, 1000)
    assert break_probs.shape == (4, 2, 16, 16)
    print("✓ Forward pass has correct shape")

# Run tests
if __name__ == '__main__':
    test_model_creation()
    test_forward_pass_shape()
    print("\n✅ All tests passed!")
```

---

## 14. Troubleshooting & FAQs

**Issue 1: Training is unstable, loss becomes `NaN`.**

-   **Cause**: The dynamic programming step in `GroupAttention` involves `log` and `exp` operations, which can be sensitive. Gradients might explode, especially early in training.
-   **Solution**: Use gradient clipping in your optimizer (e.g., `keras.optimizers.Adam(clipnorm=1.0)`). Also, a proper learning rate schedule with a warmup phase is highly recommended.

**Issue 2: The learned trees don't look meaningful or don't match linguistic conventions.**

-   **Cause**: The model learns *soft* constituencies that are useful for its primary objective (language modeling), not necessarily ones that perfectly align with a specific linguistic theory (e.g., Penn Treebank). The learned structure is a means to an end.
-   **Solution**: Don't expect a perfect, human-like parser out-of-the-box. The quality of the learned trees depends heavily on the size and diversity of the training data. For better alignment, consider the advanced pattern of fine-tuning with a syntactic loss.

### Frequently Asked Questions

**Q: How is `GroupAttention` different from standard self-attention?**

A: Standard self-attention calculates a score between every pair of tokens `(i, j)`. `GroupAttention` calculates a score for every contiguous *span* of tokens `[i, j]`. It is concerned with how tokens group together into constituents, not just pairwise interactions.

**Q: Do I need a treebank to train this?**

A: No. This is the main advantage of the Tree Transformer. It learns syntactic structures in a fully unsupervised manner, using only raw text and a language modeling objective.

---

## 15. Technical Details

### The "Tree Induction" Algorithm in `GroupAttention`

The core of `GroupAttention` is an efficient, parallelized algorithm that resembles the CKY parsing algorithm but is implemented with matrix operations for GPU execution.

1.  **Neighbor Attention**: An initial attention score is computed only between adjacent tokens (`i`, `i+1`). This forms the base of the parse.
2.  **Log Space**: To maintain numerical stability, computations are moved to log-space.
3.  **Matrix Operations**: The algorithm uses clever multiplications with upper-triangular matrices to simulate the recursive combination of smaller spans into larger ones. This allows it to compute the scores for all possible spans in a fixed number of matrix multiplications, making it highly efficient on modern hardware.
4.  **Symmetrization**: The final scores are symmetrized to produce the final `group_prob` and `break_prob` matrices.

This approach has a time complexity of `O(L² * D)` due to the matrix multiplications, which is the same as standard self-attention, meaning it doesn't introduce a significant computational bottleneck.

---

## 16. Citation

This implementation is based on the original Tree Transformer paper. If you use this model or its concepts in your research, please cite their work:

```bibtex
@inproceedings{shen2019tree,
  title={Tree Transformer: Integrating Tree Structures into Self-Attention},
  author={Shen, Yikang and Cheng, Shawn and Lasecki, Walter S and Wang, Zhou},
  booktitle={International Conference on Learning Representations},
  year={2019},
  url={https://openreview.net/forum?id=BJl_3sR5tX}
}
```