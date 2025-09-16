# Qwen3: Scalable Transformer Language Model

A production-ready implementation of the Qwen3 architecture, featuring a robust, standard decoder-only transformer stack integrated with Grouped Query Attention (GQA), SwiGLU Feed-Forward Networks, and optional Mixture of Experts (MoE) specialization, as seen in the Qwen3-Coder series.

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

| Variant | Layers | Hidden Size | Attn Heads | KV Heads | MoE Layers | Experts/Layer | Context Length | Description |
|---------|--------|-------------|------------|----------|------------|---------------|----------------|-------------|
| `30b-coder` | 48 | 2048 | 32 | 4 | 16 | 128 | 262,144 | Production MoE Coder model |
| `medium` | 24 | 1024 | 16 | 4 | 5 | 16 | 16,384 | Medium-sized MoE-enabled model |
| `small` | 12 | 768 | 12 | 4 | 3 | 8 | 4,096 | Small MoE for research/experimentation |

*Note: The number of MoE Layers indicates the indices of the 48/24/12 total transformer layers that use an MoE block instead of a standard FFN block.*

## Installation

```bash
pip install dl-techniques keras>=3.8.0 tensorflow>=2.18.0
```

## Quick Start

### Basic Usage

```python
from dl_techniques.models.qwen3.model import Qwen3

# Create a small MoE-enabled backbone model
config = Qwen3.MODEL_VARIANTS["small"].copy()
backbone = Qwen3(**config)

# Get hidden states (eager execution)
input_ids = keras.random.uniform((1, 128), 0, config["vocab_size"], dtype="int32")
hidden_states = backbone(input_ids)
print(f"Hidden State shape: {hidden_states.shape}")  # (1, 128, 768)

# Print detailed model information
backbone.summary()
```

### Text Generation

Use the factory function to create a complete model with a language modeling head.

```python
from dl_techniques.models.qwen3.model import create_qwen3

# Create optimized generation model using the 'medium' variant
model = create_qwen3("medium", task_type="generation", use_weight_tying=True)

# Compile for generation
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy"
)

# Use for inference
input_ids = keras.ops.ones((2, 64), dtype="int32")
attention_mask = keras.ops.ones((2, 64), dtype="int32")
logits = model({"input_ids": input_ids, "attention_mask": attention_mask})
```

### Classification Fine-tuning

```python
from dl_techniques.models.qwen3.model import create_qwen3

# Create classification model using the 'small' variant
classifier = create_qwen3(
    config_or_variant="small", 
    task_type="classification", 
    num_labels=5,
    pooling_strategy="cls"
)

# Compile for classification
classifier.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=2e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# classifier.fit(train_data, epochs=3, validation_data=val_data)
```

## Advanced Configuration

### Custom Model Configuration

```python
from dl_techniques.models.qwen3.model import Qwen3

# Custom configuration for a dense model (no MoE)
model = Qwen3(
    vocab_size=50000,
    hidden_size=512,
    num_layers=16,
    num_attention_heads=8,
    num_key_value_heads=2,  # Uses GQA
    intermediate_size=1376, # Typical Qwen scaling factor (512 * 2.7)
    max_seq_len=8192,
    moe_layers=[], # Dense model
    dropout_rate=0.05
)

model.summary()
```

### Manual MoE Configuration

To build a model matching the MoE setup of Qwen3, ensure `moe_layers` lists the indices where MoE replaces the standard FFN.

```python
from dl_techniques.models.qwen3.model import Qwen3

# Custom MoE setup (MoE every 4th layer)
custom_moe_model = Qwen3(
    vocab_size=32000,
    hidden_size=768,
    num_layers=20,
    num_attention_heads=12,
    num_key_value_heads=4,
    intermediate_size=2048,
    moe_layers=list(range(3, 20, 4)), # Layers 3, 7, 11, 15, 19 use MoE
    num_experts=16,
    num_experts_per_tok=4,
    moe_intermediate_size=1024, # Smaller expert size than dense FFN size
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
| **Total Parameters** | $\approx 30$ Billion (Equivalent Capacity $\approx 235$ Billion) |
| **Active Parameters** | $\approx 3$ Billion (per token) |
| **Context Length** | 262,144 tokens |
| **Inference Speed** | Optimized via GQA |
| **Training Cost** | Optimized via sparse MoE routing/load balancing |

## Training and Fine-tuning

### MoE Loss Integration

When training an MoE-enabled Qwen3 model, the auxiliary expert balancing loss must be included in the total loss function. This implementation automatically registers the expert losses within the Keras `Model.losses` property.

```python
# Custom training loop incorporating MoE auxiliary loss
import tensorflow as tf
import keras

@tf.function
def train_step(batch, model, optimizer):
    with tf.GradientTape() as tape:
        logits = model(batch, training=True)
        
        # 1. Standard Language Modeling Loss
        lm_loss = keras.losses.sparse_categorical_crossentropy(
            batch["labels"], logits, from_logits=True
        )
        mean_lm_loss = tf.reduce_mean(lm_loss)
        
        # 2. MoE Auxiliary Loss (automatically collected from the backbone)
        backbone = model.get_layer("qwen3_backbone")
        aux_loss = backbone.get_auxiliary_loss() if backbone.moe_layers else 0.0
        
        # Total Loss = LM Loss + Aux Loss * Weight (0.01)
        # Note: The weight is often pre-applied inside the MoE layer, but can be scaled here.
        total_loss = mean_lm_loss + aux_loss
    
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return {
        "total_loss": total_loss,
        "lm_loss": mean_lm_loss,
        "aux_loss": aux_loss
    }
```

## Technical Specifications

### Architecture Details

- **Block Structure**: Standard pre-norm decoder transformer.
- **Normalization**: RMS normalization ($\text{RMSNorm}(x) = x / \sqrt{\text{mean}(x^2) + \epsilon}$)
- **Position Encoding**: Rotary Position Embeddings (RoPE) applied only in attention layer.
- **FFN**: SwiGLU Gated FFN (SiLU activation).
- **Expert Networks**: SwiGLU-based feed-forward networks (for MoE layers).
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
from dl_techniques.models.qwen3.model import create_qwen3
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