# Tree Transformer: Grammar Induction with Hierarchical Attention

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A production-ready, fully-featured implementation of the **Tree Transformer** in **Keras 3**. This implementation is based on the paper "Tree Transformer: Integrating Tree Structures into Self-Attention," which introduces a novel attention mechanism capable of learning soft constituency trees from raw text without explicit syntactic supervision.

The architecture's key innovation is its **Hierarchical Group Attention** mechanism, which enables the model to infer syntactic structure as a byproduct of standard language modeling.

### Key Features
-   **Pure Foundation Model:** The `TreeTransformer` class is a pure encoder, separate from any task-specific heads.
-   **Keras 3 Native:** Built with `keras.ops` for backend-agnostic performance.
-   **Fully Serializable:** All custom layers use `@keras.saving.register_keras_serializable` for easy saving and loading with `.keras` files.
-   **Pre-built Variants:** Includes `tiny`, `small`, `base`, and `large` configurations via the `from_variant` factory method.
-   **Task Integration:** Comes with a `create_tree_transformer_with_head` factory to easily combine the encoder with heads for downstream tasks like classification or NER.

---

## Table of Contents

1.  [Overview: What is the Tree Transformer?](#1-overview-what-is-the-tree-transformer)
2.  [The Problem: The "Flat World" of Standard Attention](#2-the-problem-the-flat-world-of-standard-attention)
3.  [How It Works: Core Concepts](#3-how-it-works-core-concepts)
4.  [Architecture Deep Dive](#4-architecture-deep-dive)
5.  [Quick Start Guide](#5-quick-start-guide)
6.  [Component Reference](#6-component-reference)
7.  [Configuration & Model Variants](#7-configuration--model-variants)
8.  [Comprehensive Usage Examples](#8-comprehensive-usage-examples)
9.  [Performance Optimization](#9-performance-optimization)
10. [Serialization & Deployment](#10-serialization--deployment)
11. [Testing & Validation](#11-testing--validation)
12. [Troubleshooting & FAQs](#12-troubleshooting--faqs)
13. [Technical Details](#13-technical-details)
14. [Citation](#14-citation)

---

## 1. Overview: What is the Tree Transformer?

The **Tree Transformer** is an advanced transformer architecture that enhances self-attention with a hierarchical structure, allowing it to learn soft, latent constituency trees directly from text.

Instead of treating a sentence as a flat sequence, the Tree Transformer uses **Group Attention** to compute the probability of any given span of words forming a coherent syntactic constituent (like a noun phrase). This learned structure then modulates the standard multi-head attention, biasing the model to attend to meaningful phrases rather than arbitrary token pairs.

| Model Type | Approach | Limitations |
| :--- | :--- | :--- |
| **Traditional Parsers** | Rely on supervised training on treebanks. | Require expensive, manually annotated data not available in most languages. |
| **Standard Transformers** | Use a "flat" self-attention mechanism. | Lack an explicit inductive bias for the hierarchical nature of language. |
| **Tree Transformer** | Integrates unsupervised grammar induction into the attention mechanism. | **Combines the power of transformers with a built-in bias for syntax, learning structure and language representation simultaneously.** |

This integrated approach is critical for tasks that benefit from a deep understanding of syntax, such as grammar induction, complex question answering, and low-resource NLP.

---

## 2. The Problem: The "Flat World" of Standard Attention

Human language is inherently hierarchical. Standard self-attention, however, operates on a flat sequence, lacking a built-in bias to recognize that "the old man" is a more meaningful unit to attend to than "man who is".

```
┌─────────────────────────────────────────────────────────────┐
│  The Challenge of Missing Hierarchical Bias                 │
│                                                             │
│  Sentence: The old man who lives next door is a scientist.  │
│                                                             │
│  - Human understanding groups "The old man" and "who lives  │
│    next door" into meaningful phrases.                      │
│                                                             │
│  - Standard attention sees a flat list of tokens and must   │
│    learn these groupings from scratch, which is inefficient.│
└─────────────────────────────────────────────────────────────┘
```

The Tree Transformer explicitly introduces this missing hierarchical bias by calculating the probability that any substring forms a valid constituent and using those probabilities to guide the attention mechanism.

---

## 3. How It Works: Core Concepts

The model processes token IDs through a stack of specialized transformer blocks, outputting both language modeling predictions and the learned syntactic structure.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   Tree Transformer Complete Data Flow                   │
└─────────────────────────────────────────────────────────────────────────┘

STEP 1: EMBEDDING
─────────────────
Input IDs (B, L) ───► Embedding & PositionalEncoding ───► x (B, L, D)


STEP 2: HIERARCHICAL PROCESSING (inside one TreeTransformerBlock)
─────────────────────────────────────────────────────────────────
Inputs: x (B, L, D), group_prob_prior (from prev layer, init with 0.0)
    │
    ├─► 1. GroupAttention(x, mask, group_prob_prior)
    │      └───► Outputs: group_prob_out (B, L, L), break_prob (B, L, L)
    │
    ├─► 2. Pre-LN Residual Attention
    │      ├─► x_norm = LayerNorm(x)
    │      ├─► attn_out = TreeMHA(q, k, v, group_prob_out, mask)
    │      └───► x = x + Dropout(attn_out)
    │
    ├─► 3. Pre-LN Residual FFN
    │      ├─► x_norm = LayerNorm(x)
    │      ├─► ffn_out = FFN(x_norm)
    │      └───► x = x + Dropout(ffn_out)
    │
    └───► Outputs: x, group_prob_out (to next block), break_prob (to collect)


STEP 3: FINAL PROJECTION & OUTPUT
─────────────────────────────────
Final Hidden States (from last block)
    │
    ├─► Final LayerNorm
    │
    ├─► LM Head (Dense Layer) ───► Logits (B, L, VocabSize)
    │
    └───► All `break_prob` tensors are stacked ───► Break Probs (B, NumLayers, L, L)
```

---

## 4. Architecture Deep Dive

-   **`GroupAttention` Layer**: The core innovation. It computes **neighbor attention** between adjacent tokens and then uses a dynamic programming-like algorithm (implemented with matrix multiplications) to efficiently compute scores for all possible spans. The output is a matrix representing the likelihood of each span being a single constituent.
-   **`TreeMHA` Layer**: A modified multi-head attention layer. It performs standard scaled dot-product attention and then multiplies the attention weights element-wise with the probabilities from `GroupAttention`, biasing the model to focus on learned constituents.
-   **`TreeTransformerBlock`**: A self-contained block following a Pre-LN design for stability. It orchestrates the flow: `GroupAttention` → `TreeMHA` → `FFN`, with residual connections and layer normalization. It also passes the computed group probabilities to the next block as a prior.

---

## 5. Quick Start Guide

### Your First Tree Transformer Model (30 seconds)

This example builds a `tiny` model and inspects its outputs for a dummy sequence.

```python
import keras
import numpy as np
from model import TreeTransformer

# 1. Create a "tiny" Tree Transformer from a pre-defined variant
# This model is a pure encoder, ready to be used as a foundation.
model = TreeTransformer.from_variant("tiny", vocab_size=5000)
model.summary()
print("✅ Tree Transformer foundation model created successfully!")

# 2. Generate a batch of dummy input data (integer token IDs)
batch_size, seq_len = 4, 32
dummy_input_ids = keras.ops.convert_to_tensor(
    np.random.randint(1, 5000, size=(batch_size, seq_len)), dtype="int32"
)

# 3. Perform a forward pass
# The model expects a dictionary and returns a dictionary of outputs.
inputs = {"input_ids": dummy_input_ids}
outputs = model(inputs, training=False)

# 4. Inspect the shapes of the outputs
print(f"\nOutput keys: {list(outputs.keys())}")
print(f"Last hidden state shape: {outputs['last_hidden_state'].shape}")
print(f"Logits shape: {outputs['logits'].shape}")
print(f"Break probabilities shape: {outputs['break_probs'].shape}")
```

---

## 6. Component Reference

| Component | Type | Purpose |
| :--- | :--- | :--- |
| `TreeTransformer` | `keras.Model` | The main foundation model. A pure encoder that returns hidden states, logits, and break probabilities. |
| `TreeTransformer.from_variant()` | Class Method | The recommended factory for creating `TreeTransformer` instances with standard configurations like "tiny", "base", etc. |
| `TreeTransformerBlock` | `keras.layers.Layer` | A single layer of the encoder, containing `GroupAttention`, `TreeMHA`, and an `FFN`. |
| `GroupAttention` | `keras.layers.Layer` | The core layer that computes unsupervised constituency probabilities. |
| `TreeMHA` | `keras.layers.Layer` | A multi-head attention layer whose scores are modulated by the output of `GroupAttention`. |
| `create_tree_transformer_with_head()` | Function | A factory to build a complete end-to-end model by combining a `TreeTransformer` with a task-specific head (e.g., for NER). |

---

## 7. Configuration & Model Variants

Create models of different sizes using the `from_variant` method. You can override any default parameter.

| Variant | `hidden_size` | `num_layers` | `num_heads` | `intermediate_size` |
| :--- | :---: | :---: | :---: | :---: |
| `tiny` | 128 | 4 | 4 | 512 |
| `small` | 256 | 6 | 4 | 1024 |
| `base` | 512 | 10 | 8 | 2048 |
| `large` | 1024 | 16 | 16 | 4096 |

**Example: Creating a `base` model with a custom vocabulary size**
```python
model = TreeTransformer.from_variant(
    "base",
    vocab_size=50257,
    max_len=512
)
```

---

## 8. Comprehensive Usage Examples

### Example 1: Setting up for Masked Language Modeling (MLM)

The primary training objective for Tree Transformer is MLM, identical to BERT.

```python
import keras
from model import TreeTransformer

# 1. Create the model
model = TreeTransformer.from_variant("base", vocab_size=30000)

# 2. Define optimizer and loss
# Keras's standard cross-entropy loss handles masked labels automatically
# if the ignored positions are set to a negative value (e.g., -100).
optimizer = keras.optimizers.Adam(learning_rate=5e-5)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# 3. Compile the model
# We only care about the 'logits' output for the MLM loss.
model.compile(optimizer=optimizer, loss=loss_fn, weighted_metrics=["accuracy"])

# 4. Train the model (assuming a data generator `mlm_dataset`)
# The dataset should yield dictionaries:
# inputs = {"input_ids": masked_tokens}
# targets = original_tokens_with_-100_for_non_masked
# model.fit(mlm_dataset, epochs=3)
```

### Example 2: Fine-tuning for a Downstream Task (NER)

Use the `create_tree_transformer_with_head` factory to easily build a model for any NLP task.

```python
import keras
from model import create_tree_transformer_with_head
from dl_techniques.nlp.heads.task_types import NLPTaskConfig, NLPTaskType

# 1. Define the task configuration
ner_config = NLPTaskConfig(
    name="ner",
    task_type=NLPTaskType.NAMED_ENTITY_RECOGNITION,
    num_classes=9  # e.g., B-PER, I-PER, B-LOC, etc. + O
)

# 2. Create the complete model
# This function handles creating the TreeTransformer base and adding the NER head.
ner_model = create_tree_transformer_with_head(
    tree_transformer_variant="base",
    task_config=ner_config,
    encoder_config_overrides={"vocab_size": 30000}
)
ner_model.summary()

# 3. Compile and fine-tune
ner_model.compile(
    optimizer=keras.optimizers.Adam(3e-5),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)
# ner_model.fit(ner_dataset, epochs=3)
```

### Example 3: Interpreting the Learned Tree

The `break_probs` output can be used to visualize the learned syntactic structure. A high value at `break_probs[i, i+1]` suggests a syntactic boundary.

```python
import matplotlib.pyplot as plt

# Get outputs from the quick start example
outputs = model(inputs, training=False)
break_probs = outputs["break_probs"]

# Visualize group probabilities (1 - break_prob) from the last layer for one sentence
group_probs = 1.0 - break_probs
last_layer_group_probs = group_probs[0, -1, :, :] # (seq_len, seq_len)

plt.figure(figsize=(10, 8))
plt.imshow(keras.ops.convert_to_numpy(last_layer_group_probs), cmap='viridis')
plt.colorbar(label="Constituency Score (P(span))")
plt.title("Learned Group Attention Matrix (Last Layer, Sample 0)")
plt.xlabel("Token Index")
plt.ylabel("Token Index")
plt.show()
```

---

## 9. Performance Optimization

-   **Mixed Precision Training**: For larger models (`base` or `large`), mixed precision can significantly speed up training on compatible GPUs.
    ```python
    keras.mixed_precision.set_global_policy('mixed_float16')
    ```
-   **XLA Compilation**: Use `jit_compile=True` during `model.compile()` for graph compilation, which can provide a speed boost for the matrix-heavy operations in `GroupAttention`.
    ```python
    model.compile(..., jit_compile=True)
    ```

---

## 10. Serialization & Deployment

All custom layers are registered with Keras's serialization system, making saving and loading seamless.

### Saving and Loading

```python
# Create and train model
model = TreeTransformer.from_variant("tiny", vocab_size=5000)
# model.compile(...) and model.fit(...)

# Save the entire model (weights, config, optimizer state)
model.save('my_tree_transformer.keras')

# Load the model in a new session. No custom_objects needed for the layers.
loaded_model = keras.models.load_model('my_tree_transformer.keras')
loaded_model.summary()
```

---

## 11. Testing & Validation

This implementation includes a comprehensive `pytest` suite in `test_model.py`, covering:
-   Model initialization and parameter validation.
-   Correctness of output shapes for all model variants.
-   Serialization and deserialization consistency.
-   End-to-end integration with task heads, including gradient flow checks.
-   Edge cases like padding and minimum sequence length.

---

## 12. Troubleshooting & FAQs

**Issue 1: Training is unstable, loss becomes `NaN`.**
-   **Cause**: The `log` and `exp` operations in `GroupAttention` can be sensitive.
-   **Solution**: Use gradient clipping in your optimizer (e.g., `keras.optimizers.Adam(clipnorm=1.0)`) and a learning rate schedule with a warmup phase.

**Issue 2: The learned trees don't match linguistic conventions.**
-   **Cause**: The model learns *soft* constituencies that are useful for its primary objective (language modeling), not necessarily ones that perfectly align with a specific linguistic theory.
-   **Solution**: Interpret the trees as a representation of the model's internal grouping strategy. For better alignment with a specific treebank, consider fine-tuning with a secondary syntactic loss.

**Q: Do I need a treebank to train this?**
A: **No.** This is the main advantage. It learns syntax in a fully unsupervised manner from raw text.

---

## 13. Technical Details

The core of `GroupAttention` is an efficient algorithm resembling CKY parsing, but implemented with matrix operations for GPU execution. It uses clever multiplications with upper-triangular matrices to simulate the recursive combination of smaller spans into larger ones. This allows it to compute scores for all spans in a fixed number of matrix multiplications, maintaining a time complexity of `O(L² * D)`, the same as standard self-attention.

---

## 14. Citation

This implementation is based on the original Tree Transformer paper. If you use this model in your research, please cite their work:

```bibtex
@inproceedings{shen2019tree,
  title={Tree Transformer: Integrating Tree Structures into Self-Attention},
  author={Shen, Yikang and Cheng, Shawn and Lasecki, Walter S and Wang, Zhou},
  booktitle={International Conference on Learning Representations},
  year={2019},
  url={https://openreview.net/forum?id=BJl_3sR5tX}
}
```