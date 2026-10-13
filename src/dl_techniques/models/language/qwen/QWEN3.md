# Qwen3: Scalable Transformer Language Model

An implementation of the Qwen3 architecture, featuring a robust, standard decoder-only transformer stack integrated with Grouped Query Attention (GQA), SwiGLU Feed-Forward Networks, and optional Mixture of Experts (MoE) specialization, as seen in the Qwen3-Coder series.

## Overview

The Qwen3 architecture is designed for high performance and scalability, combining proven transformer components with modern optimizations like Grouped Query Attention for inference acceleration and Sparse Mixture of Experts for massive capacity expansion.

This implementation, built using the `dl_techniques` framework, provides a modular backbone suitable for both dense (FFN) and sparse (MoE) configurations.

### Key Architecture Features

- **Standard Decoder Block**: Classical pre-normalization transformer layers.
- **Grouped Query Attention (GQA)**: Accelerates large-scale inference by sharing K/V heads among groups of Q heads.
- **RMS Normalization**: Uses RMSNorm (pre-norm configuration) for enhanced training stability.
- **SwiGLU FFN**: Replaces standard ReLU/GeLU FFNs with the computationally efficient Gated Linear Unit (GLU) variant using SiLU activation.
- **Optional Mixture of Experts (MoE)**: Enables scaling model capacity by replacing standard FFN layers with sparse MoE layers in specific variants (e.g., Coder models).

## Architecture Diagram

The fundamental structure of the Qwen3 backbone (non-MoE/Hybrid layers):

```
Input Processing:
    Token IDs (vocab_size=...) → Token Embeddings → RoPE Position Encoding
                                        ↓
Transformer Layer (repeated N times):
    ┌─ RMSNorm → GQA (Attention) → ⊕ (Residual)
    └─ RMSNorm → SwiGLU FFN → ⊕ (Residual) 
                                        ↓
Output Processing:
    Final RMSNorm → Linear Projection → Logits (vocab_size=...)
```

In MoE variants (like `30b-coder`), the FFN step in selected layers is replaced by a Sparse MoE layer.

### Detailed Layer Structure

Each Qwen3 block utilizes the pre-normalization residual block structure:
1. **Input Normalization**: RMSNorm applied to the input of the layer.
2. **Core Layer**: Either GQA or SwiGLU/MoE is applied.
3. **Dropout**
4. **Residual Connection**: Output is added back to the pre-normalized input.

**Grouped Query Attention (GQA) Features:**
- Reduces KV cache size compared to standard Multi-Head Attention (MHA).
- Rotational Position Embeddings (RoPE) are applied to Q and K vectors.
- Optimized for speed on modern accelerators.

**SwiGLU FFN:**
- Uses the `(X @ W_gate) ⊙ SiLU(X @ W_up)` structure, improving performance and accuracy over standard FFNs.

## Model Variants

| Variant | Layers | Hidden Size | Attn Heads | KV Heads | MoE Layers | Experts/Layer | `moe_intermediate_size` | Trainable parameters (built) | Context Length | Description |
|---------|--------|-------------|------------|----------|------------|---------------|-------------------------|--------------------|----------------|-------------|
| `30b-coder` | 48 | 2048 | 32 | 4 | 16 | 128 | 768 | 11.85B | 262,144 | Production MoE Coder model |
| `medium` | 24 | 1024 | 16 | 4 | 5 | 16 | 2048 | 936M | 16,384 | Medium-sized MoE-enabled model |
| `small` | 12 | 768 | 12 | 4 | 3 | 8 | 1024 | 167M | 4,096 | Small MoE for research/experimentation |
| `tiny` | 6 | 512 | 8 | 2 | 0 | 1 | 2048 (unread) | 50.9M | 2,048 | Dense, no MoE; tests and examples |

*Note: The number of MoE Layers counts the indices, out of the 48/24/12/6 transformer layers, that use an MoE block instead of a standard FFN block. `moe_intermediate_size` is each expert's FFN hidden width, used verbatim (the Hugging Face Qwen3-MoE meaning); dense layers do not read it (their SwiGLU is sized from `4 * hidden_size` by its 2/3 rule, rounded up to a multiple of 256: 1536 at hidden 512, 2048 at 768, 2816 at 1024, 5632 at 2048), so for `tiny` it is unread. "Trainable parameters (built)" counts the trainable weights of what this implementation builds, excluding the non-trainable RoPE `cos_cached`/`sin_cached` buffers (805M of `count_params()` for `30b-coder`, 25.2M for `medium`, 3.1M for `small`, 0.8M for `tiny`). Measured on CPU: exact for `medium`, `small` and `tiny`; `30b-coder` extrapolated from 1- and 2-layer dense builds and 2- and 3-expert MoE builds, a method that reproduces the three exact counts. Before 2026-10-13 (D-014) an expert was a SwiGLU sized by `max(1, moe_intermediate_size // hidden_size)` through the 2/3 rule (30b-coder 1536 wide, medium 1536, small 512) and the trainable counts were 21.52B, 810M and 139M; MoE archives of those shapes are refused by `from_variant(pretrained=...)` and by `keras.models.load_model`.*

## Installation

```bash
pip install dl-techniques keras>=3.8.0 tensorflow>=2.18.0
```

## Quick Start

### Basic Usage

```python
import keras
from dl_techniques.models.language.qwen.qwen3 import Qwen3

# Create the small MoE-enabled model (with its LM head)
backbone = Qwen3.from_variant("small")

# The model returns vocabulary logits (eager execution)
input_ids = keras.random.randint((1, 128), 0, 32000, dtype="int32")
logits = backbone(input_ids)
print(f"Logits shape: {logits.shape}")  # (1, 128, 32000)

# Print detailed model information
backbone.summary()
```

### Text Generation

`create_qwen3(variant, pretrained=False, **kwargs)` forwards to `Qwen3.from_variant` and
returns the `Qwen3` itself; without a task head it is the language model.

```python
import keras
from dl_techniques.models.language.qwen.qwen3 import create_qwen3

# A language model from the 'tiny' variant
model = create_qwen3("tiny")

# Compile for next-token training (the model returns logits)
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)

# Use for inference
input_ids = keras.ops.ones((2, 64), dtype="int32")
attention_mask = keras.ops.ones((2, 64), dtype="int32")
logits = model({"input_ids": input_ids, "attention_mask": attention_mask})  # (2, 64, 32000)
```

### Classification Fine-tuning

A classifier is the same class with a task head, set through `include_top=True` and a
`task_config`. The head pools the last position kept by `attention_mask` by default;
`head_config_overrides={"pooling_type": ...}` changes that.

```python
import keras
from dl_techniques.layers.heads.nlp import NLPTaskConfig, NLPTaskType
from dl_techniques.models.language.qwen.qwen3 import create_qwen3

# Create a 5-class classifier from the 'small' variant
classifier = create_qwen3(
    "small",
    include_top=True,
    task_config=NLPTaskConfig(
        name="sentiment", task_type=NLPTaskType.TEXT_CLASSIFICATION, num_classes=5),
)

# Compile for classification (the head returns logits)
classifier.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=2e-5),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# classifier.fit({"input_ids": ids, "attention_mask": mask}, labels, epochs=3)
```

## Advanced Configuration

### Custom Model Configuration

```python
from dl_techniques.models.language.qwen.qwen3 import Qwen3

# Custom configuration for a dense model (no MoE)
model = Qwen3(
    vocab_size=50000,
    hidden_size=512,
    num_layers=16,
    num_attention_heads=8,
    num_key_value_heads=2,  # Uses GQA
    max_seq_len=8192,
    moe_layers=[], # Dense model
    dropout_rate=0.05
)

model.build((None, None))
model.summary()
```

### Manual MoE Configuration

To build a model matching the MoE setup of Qwen3, ensure `moe_layers` lists the indices where MoE replaces the standard FFN.

```python
from dl_techniques.models.language.qwen.qwen3 import Qwen3

# Custom MoE setup (MoE every 4th layer)
custom_moe_model = Qwen3(
    vocab_size=32000,
    hidden_size=768,
    num_layers=20,
    num_attention_heads=12,
    num_key_value_heads=4,
    moe_layers=list(range(3, 20, 4)), # Layers 3, 7, 11, 15, 19 use MoE
    num_experts=16,
    num_experts_per_tok=4,
    moe_intermediate_size=1024, # Hidden width of each expert's FFN
    rope_theta=10000.0,
)
```

## Performance Characteristics

### Computational Efficiency

Qwen3 leverages Grouped Query Attention (GQA) and sparse MoE layers (in specific variants) to optimize resource usage, especially during inference.

- **Inference Speed**: GQA significantly reduces memory bandwidth requirements for the KV cache compared to traditional MHA, accelerating inference for long sequences.
- **Sparse Activation**: In MoE variants, only $k$ (e.g., 8) out of $E$ (e.g., 128) experts are active per token, maintaining a high capacity without a prohibitive computational load.

### Resource Requirements (Example: `30b-coder` MoE variant)

| Metric | Detail |
|---------|--------------------------------|
| **Total Parameters** | 11.85B trainable, built (16 of the 48 layers are MoE; the released Qwen3-Coder-30B-A3B makes every layer MoE; RoPE buffers excluded) |
| **Active Parameters** | 2.80B trainable, built, per token (8 of 128 experts in each MoE layer; router, attention, dense layers, embeddings and `lm_head` in full; RoPE buffers excluded) |
| **Context Length** | 262,144 tokens |
| **Inference Speed** | Optimized via GQA |
| **Training Cost** | Optimized via sparse MoE routing/load balancing |

## Training and Fine-tuning

### MoE Loss Integration

When training an MoE-enabled Qwen3 model, the auxiliary expert balancing loss must be included in the total loss function. The MoE layers register it through `add_loss`, so after a forward pass it is in the model's `losses` property (`fit()` adds it on its own).

```python
# Custom training loop incorporating MoE auxiliary loss
import numpy as np
import tensorflow as tf
import keras
from dl_techniques.models.language.qwen.qwen3 import Qwen3

def train_step(batch, model, optimizer):
    with tf.GradientTape() as tape:
        logits = model(batch["input_ids"], training=True)

        # 1. Standard Language Modeling Loss
        lm_loss = keras.losses.sparse_categorical_crossentropy(
            batch["labels"], logits, from_logits=True
        )
        mean_lm_loss = tf.reduce_mean(lm_loss)

        # 2. MoE Auxiliary Loss (collected from the MoE layers by the forward pass)
        aux_loss = tf.add_n(model.losses) if model.losses else 0.0

        total_loss = mean_lm_loss + aux_loss

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return {
        "total_loss": total_loss,
        "lm_loss": mean_lm_loss,
        "aux_loss": aux_loss
    }

# One step on a small MoE model
model = Qwen3(vocab_size=1000, hidden_size=64, num_layers=2, num_attention_heads=4,
              num_key_value_heads=2, max_seq_len=64, moe_layers=[1], num_experts=4,
              num_experts_per_tok=2, moe_intermediate_size=64)
ids = np.random.randint(1, 1000, (2, 16)).astype("int32")
step = train_step({"input_ids": ids, "labels": ids}, model, keras.optimizers.AdamW(1e-4))
```

## Technical Specifications

### Architecture Details

- **Block Structure**: Standard pre-norm decoder transformer.
- **Normalization**: RMS normalization ($\text{RMSNorm}(x) = x / \sqrt{\text{mean}(x^2) + \epsilon}$)
- **Position Encoding**: Rotary Position Embeddings (RoPE) applied only in attention layer.
- **FFN**: SwiGLU Gated FFN (SiLU activation), sized from `4 * hidden_size` by the SwiGLU 2/3 rule.
- **Expert Networks**: SwiGLU-based feed-forward networks (for MoE layers), each `moe_intermediate_size` wide.
- **Attention**: Grouped Query Attention (GQA).

### Mathematical Foundation

**Attention Calculation:**
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q \cdot K^T}{\sqrt{d_k}} + M\right) \cdot V
$$
Where $M$ is the causal mask combined with the padding mask. $Q$ and $K$ are processed by RoPE.

**SwiGLU FFN:**
$$
\text{SwiGLU}(x) = (\text{SiLU}(x \cdot W_1) \odot (x \cdot W_2)) \cdot W_3
$$

**Mixture of Experts Routing:**
$$
\text{Gate}(x) = \text{TopK}_{k}\left(\text{Softmax}(x \cdot W_{\text{gate}})\right)
$$
$$
\text{Output}(x) = \sum_{i \in \text{TopK}} \text{Gate}_i(x) \cdot \text{Expert}_i(x)
$$

## Model Serialization

```python
from dl_techniques.models.language.qwen.qwen3 import create_qwen3
import keras

# Save model (includes MoE components if present)
model = create_qwen3("small")
model.save("qwen3_small_generation.keras")

# Load model (automatic layer registration)
loaded_model = keras.models.load_model("qwen3_small_generation.keras")

# Verify architecture
loaded_model.summary()
```

---