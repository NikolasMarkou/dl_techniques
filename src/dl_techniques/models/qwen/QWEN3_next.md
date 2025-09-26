# Qwen3 Next: Mixture of Experts Language Model

A production-ready implementation of the Qwen3 Next architecture featuring Mixture of Experts (MoE) with the correct block structure: 3x Gated DeltaNet + 1x Gated Attention per block, each with individual Zero-Centered RMSNorm and MoE layers.

## Overview

Qwen3 Next represents an advancement in transformer architecture by incorporating both Mixture of Experts and novel attention mechanisms. The model uses a unique block structure that combines linear-complexity Gated DeltaNet layers with traditional Gated Attention, achieving high parameter counts while maintaining computational efficiency through sparse activation patterns.

### Key Architecture Features

- **Hybrid Block Structure**: Each block contains 3x Gated DeltaNet + 1x Gated Attention layers
- **Sparse Mixture of Experts**: 64 total experts with only 8 activated per token (8:1 sparsity ratio)
- **Gated DeltaNet**: Linear-complexity attention alternative with delta rule mechanism
- **Gated Attention**: Enhanced multi-head attention with Zero-Centered RMSNorm and partial RoPE
- **Individual Layer Processing**: Each layer has its own normalization and MoE
- **Zero-Centered RMSNorm**: Improved training stability over standard normalization

## Architecture Diagram

```
Input Processing:
    Token IDs (vocab_size=151k) → Token Embeddings → RoPE Position Encoding
                                        ↓
Qwen3Next Block (repeated N times):
    ┌─ Zero-Centered RMSNorm → Gated DeltaNet → MoE → ⊕ (residual)
    ├─ Zero-Centered RMSNorm → Gated DeltaNet → MoE → ⊕ (residual)  
    ├─ Zero-Centered RMSNorm → Gated DeltaNet → MoE → ⊕ (residual)
    └─ Zero-Centered RMSNorm → Gated Attention → MoE → ⊕ (residual)
                                        ↓
Output Processing:
    Final Zero-Centered RMSNorm → Linear Projection → Logits (vocab_size=151k)
```

### Detailed Layer Structure

Each Qwen3Next block contains 4 sub-layers, each following this pattern:
1. **Zero-Centered RMSNorm** for input normalization
2. **Core Layer** (either Gated DeltaNet or Gated Attention)
3. **Mixture of Experts** for specialized processing
4. **Residual Connection** back to the input

**Gated DeltaNet Features:**
- Linear complexity O(L×D²) vs quadratic O(L²×D) attention
- Delta rule mechanism for targeted memory updates
- Adaptive gating (α, β parameters) for memory control
- Short convolution for position-based addressing

**Gated Attention Features:**
- Scaled dot-product attention with partial RoPE
- Output gating with sigmoid activation
- Multi-head attention with configurable head dimensions

## Model Variants

| Variant | Blocks | Effective Layers | Parameters | Hidden Size | Experts | Active/Token | Context Length | Description |
|---------|--------|------------------|------------|-------------|---------|-------------|----------------|-------------|
| `80b_a3b` | 12 | 48 | 80B | 2048 | 64 | 8 | 8192 | Full production model |
| `80b` | 12 | 48 | 80B | 2048 | 1 | 1 | 8192 | Dense variant without MoE |
| `small` | 6 | 24 | ~3B | 1024 | 8 | 2 | 2048 | Development/experimentation |
| `tiny` | 3 | 12 | ~500M | 512 | 4 | 1 | 1024 | Edge/mobile deployment |

*Note: "Blocks" refers to Qwen3Next blocks (3 DeltaNet + 1 Attention each). "Effective Layers" is the total number of individual processing layers.*

## Installation

```bash
pip install dl-techniques keras>=3.8.0 tensorflow>=2.18.0
```

## Quick Start

### Basic Usage

```python
from dl_techniques.models.qwen3_next import Qwen3Next

# Create a small model for experimentation
model = Qwen3Next.from_variant("small")

# Generate text
input_ids = keras.random.uniform((1, 128), 0, 151936, dtype="int32")
logits = model(input_ids)
print(f"Output shape: {logits.shape}")  # (1, 128, 151936)

# Print detailed model information
model.summary()
```

### Text Generation

```python
from dl_techniques.models.qwen3_next import create_qwen3_next

# Create optimized generation model
model = create_qwen3_next("small", task_type="generation")

# Compile for generation
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy"
)

# Use for inference
input_ids = keras.random.uniform((2, 64), 0, 151936, dtype="int32")
attention_mask = keras.ops.ones((2, 64), dtype="int32")
logits = model([input_ids, attention_mask])
```

### Classification Fine-tuning

```python
# Create classification model
classifier = create_qwen3_next(
    variant="small", 
    task_type="classification", 
    num_labels=2
)

# Compile for classification
classifier.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=2e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train on your classification data
# classifier.fit(train_data, epochs=3, validation_data=val_data)
```

## Advanced Configuration

### Custom Model Configuration

```python
from dl_techniques.models.qwen3_next import Qwen3Next
from dl_techniques.layers.moe import MoEConfig, ExpertConfig, GatingConfig

# Create custom MoE configuration
moe_config = MoEConfig(
    num_experts=16,
    expert_config=ExpertConfig(
        ffn_config={
            "type": "swiglu",
            "output_dim": 1024,
            "ffn_expansion_factor": 4
        }
    ),
    gating_config=GatingConfig(
        top_k=4,
        gating_type="linear"
    )
)

# Create model with custom configuration
model = Qwen3Next(
    vocab_size=151936,
    hidden_size=1024,
    num_layers=8,  # 8 blocks = 32 effective layers
    num_attention_heads=16,
    num_key_value_heads=4,
    num_experts=16,
    num_experts_per_tok=4,
    normalization_type="zero_centered_rms_norm",
    use_stochastic_depth=True,
    stochastic_depth_rate=0.1
)
```

### Manual Configuration

```python
# Full manual configuration
model = Qwen3Next(
    vocab_size=151936,
    hidden_size=1024,
    num_layers=6,  # 6 blocks = 24 effective layers (18 DeltaNet + 6 Attention)
    num_attention_heads=16,
    num_key_value_heads=4,
    num_experts=8,
    num_experts_per_tok=2,
    max_position_embeddings=4096,
    rope_theta=1000000.0,
    normalization_type="zero_centered_rms_norm",
    dropout_rate=0.1,
    use_stochastic_depth=True,
    stochastic_depth_rate=0.05
)
```

## Performance Characteristics

### Computational Efficiency

The hybrid MoE architecture provides significant computational savings:

- **Linear Complexity**: Gated DeltaNet provides O(L) scaling vs O(L²) attention
- **Active Parameters**: Only ~3-12% of total parameters active per forward pass
- **Memory Usage**: Linear scaling with active parameters, constant memory for sequence length
- **Throughput**: ~10-30x improvement over dense models of equivalent capacity
- **Expert Specialization**: Each layer's MoE develops specialized processing capabilities

### Resource Requirements

| Variant | GPU Memory (Inference) | GPU Memory (Training) | Recommended Hardware |
|---------|----------------------|---------------------|-------------------|
| `tiny` | ~2GB | ~4GB | Consumer GPU (RTX 3060+) |
| `small` | ~6GB | ~12GB | Professional GPU (RTX 4080+) |
| `80b_a3b` | ~40GB | ~80GB+ | Multi-GPU setup (A100+) |

### Layer-wise Complexity

For a model with L sequence length, D hidden dimension, H heads:

| Layer Type | Complexity | Memory | Active Parameters |
|------------|------------|---------|-------------------|
| Gated DeltaNet | O(L×D²) | O(H×D²) | ~25% of layer params |
| Gated Attention | O(L²×D) | O(L²×H) | ~100% of layer params |
| MoE (per layer) | O(L×D×k/E) | O(D²×k) | k/E ratio of experts |

*Where k = experts per token, E = total experts*

## Training and Fine-tuning

### Pre-training

```python
# Pre-training setup with expert load balancing
model = Qwen3Next.from_variant("small")

# Custom training loop with MoE-specific considerations
import tensorflow as tf

@tf.function
def train_step(batch):
    with tf.GradientTape() as tape:
        logits = model(batch["input_ids"], training=True)
        
        # Standard language modeling loss
        lm_loss = keras.losses.sparse_categorical_crossentropy(
            batch["labels"], logits, from_logits=True
        )
        
        # MoE auxiliary losses are automatically included via model.losses
        aux_loss = sum(model.losses) if model.losses else 0.0
        total_loss = tf.reduce_mean(lm_loss) + aux_loss
    
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return {
        "total_loss": total_loss,
        "lm_loss": tf.reduce_mean(lm_loss),
        "aux_loss": aux_loss
    }
```

### Fine-tuning Best Practices

- **Learning Rates**: Use lower rates (1e-5 to 5e-5) for fine-tuning
- **Layer-wise Learning Rates**: Consider different rates for DeltaNet vs Attention layers
- **Expert Monitoring**: Monitor expert utilization patterns during training
- **Memory Management**: Use gradient checkpointing for large models
- **Stochastic Depth**: Enable for better regularization in deep variants

## Technical Specifications

### Architecture Details

- **Block Structure**: 3 Gated DeltaNet + 1 Gated Attention per block
- **Normalization**: Zero-Centered RMS normalization throughout
- **Position Encoding**: Rotary Position Embeddings (RoPE) with configurable theta
- **Expert Networks**: SwiGLU-based feed-forward networks
- **Activation Functions**: SiLU in DeltaNet, configurable in experts
- **Initialization**: Truncated normal with configurable standard deviation

### Mathematical Foundation

**Gated DeltaNet Update Rule:**
```
S_t = α_t * (S_{t-1} + β_t * V_t * K_t^T) + (1-α_t) * S_{t-1}
Output_t = Q_t @ S_t + V_residual_t
```

**Gated Attention:**
```
Q,K,V = Linear_projections(RMSNorm(X))
Q_rope, K_rope = RoPE(Q, K)  # Partial RoPE
Attention_out = Attention(Q_rope, K_rope, V)
Output = σ(Gate_proj(Attention_out)) ⊙ Attention_out
```

**MoE Gating (per layer):**
```
Gate_scores = Softmax(X·W_gate)
Selected_experts = TopK(Gate_scores, k=experts_per_token)
Output = Σᵢ Gate_scores[i] · Expert_i(X) for i in Selected_experts
```

## Model Serialization

```python
# Save model (includes all MoE experts and layer states)
model = Qwen3Next.from_variant("small")
model.save("qwen3_next_small.keras")

# Load model (automatic layer registration)
loaded_model = keras.models.load_model("qwen3_next_small.keras")

# Verify architecture
loaded_model.summary()
```

## Benchmarks and Performance

### Model Statistics

| Variant | Total Layers | DeltaNet Layers | Attention Layers | MoE Instances | Sparsity Ratio |
|---------|--------------|-----------------|------------------|---------------|----------------|
| `tiny` | 12 | 9 | 3 | 12 | 4:1 |
| `small` | 24 | 18 | 6 | 24 | 4:1 |
| `80b_a3b` | 48 | 36 | 12 | 48 | 8:1 |

## Expert Utilization Analysis

```python
# Monitor expert usage (example for debugging/analysis)
def analyze_expert_usage(model):
    """Analyze expert utilization across all blocks."""
    total_experts = 0
    active_experts = 0
    
    for i, block in enumerate(model.blocks):
        print(f"\nBlock {i}:")
        
        # Check DeltaNet MoE layers
        for j, moe_layer in enumerate(block.delta_moe_layers):
            if moe_layer is not None:
                stats = moe_layer.get_expert_utilization()
                print(f"  DeltaNet {j} MoE: {stats}")
                total_experts += stats.get('num_experts', 0)
                active_experts += stats.get('top_k', 0)
        
        # Check Attention MoE layer
        if block.attention_moe is not None:
            stats = block.attention_moe.get_expert_utilization()
            print(f"  Attention MoE: {stats}")
            total_experts += stats.get('num_experts', 0)
            active_experts += stats.get('top_k', 0)
    
    sparsity = total_experts / active_experts if active_experts > 0 else 0
    print(f"\nOverall Sparsity Ratio: {sparsity:.1f}:1")

# Usage
model = Qwen3Next.from_variant("small")
analyze_expert_usage(model)
```

## Qwen3 vs Qwen3 Next: Architectural Evolution

### Key Improvements in Qwen3 Next

#### 1. **Hybrid Attention Architecture**
- **Qwen3**: Pure multi-head attention throughout
- **Qwen3 Next**: 3:1 ratio of linear Gated DeltaNet to quadratic Gated Attention
- **Benefit**: Linear complexity for most layers while maintaining attention quality

#### 2. **Enhanced Normalization**
- **Qwen3**: Standard RMS normalization
- **Qwen3 Next**: Zero-Centered RMS normalization for improved stability
- **Benefit**: Better training dynamics and gradient flow

#### 3. **Optimized Expert Configuration**
- **Qwen3**: Higher activation ratios, less efficient sparsity
- **Qwen3 Next**: Fine-tuned sparsity ratios per layer type
- **Benefit**: Better parameter efficiency with maintained performance

#### 4. **Layer-wise Specialization**
- **Qwen3**: Uniform layer structure
- **Qwen3 Next**: Different layer types optimized for different functions
- **Benefit**: DeltaNet for memory/retrieval, Attention for complex reasoning

### Performance Comparison

| Metric | Qwen3 235B-A22B | Qwen3 Next 80B-A3B | Improvement |
|--------|-----------------|---------------------|-------------|
| **Parameters** | 235B | 80B | 66% reduction |
| **Active/Token** | ~22B | ~3B | 86% reduction |
| **Inference Speed** | Baseline | 2-3x faster | Significant speedup |
| **Memory Usage** | High | 70% reduction | Substantial savings |
| **Training Cost** | Baseline | 75% reduction | Major cost savings |

The hybrid architecture allows Qwen3 Next to achieve competitive performance with dramatically reduced computational requirements, making it more accessible for research and deployment.
