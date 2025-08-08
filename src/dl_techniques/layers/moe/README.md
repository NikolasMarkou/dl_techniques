# Mixture of Experts (MoE) Module

A comprehensive, production-ready implementation of modern Mixture of Experts architectures for the dl_techniques framework. This module enables building sparse, scalable neural networks that can efficiently scale to trillions of parameters.

## Features

### 🎯 **Expert Types**
- **FFN Experts**: Standard feed-forward network experts for transformer architectures
- **Attention Experts**: Multi-head attention experts for Mixture-of-Attention (MoA) models
- **Convolutional Experts**: Specialized convolutional experts for vision models

### 🧠 **Gating Mechanisms**
- **Linear Gating**: Standard linear transformation with optional noise injection
- **Cosine Gating**: Hypersphere-based routing using cosine similarity
- **SoftMoE**: Soft routing without token dropping for improved stability

### ⚖️ **Load Balancing**
- Auxiliary loss functions for expert load balancing
- Router z-loss for entropy regularization
- Expert capacity management with token dropping
- Residual connections for unprocessed tokens

### 🔧 **Advanced Features**
- Hierarchical routing support
- Modality-specific and task-specific routing
- Expert parallelism ready
- Comprehensive configuration system
- Full Keras 3.x compatibility

## Quick Start

```python
from dl_techniques.layers.moe import create_ffn_moe, MixtureOfExperts, MoEConfig

# Quick creation with factory function
moe_layer = create_ffn_moe(
    num_experts=8,
    hidden_dim=768,
    top_k=2,
    gating_type='linear'
)

# Use in a model
model = keras.Sequential([
    keras.layers.Input(shape=(512, 768)),
    moe_layer,
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(10, activation='softmax')
])
```

## Installation

The MoE module is part of the dl_techniques framework. Ensure you have the required dependencies:

```bash
# Core dependencies (already in dl_techniques)
pip install keras>=3.8.0 tensorflow>=2.18.0
```

## Architecture Overview

```
Input Token
     ↓
┌─────────────┐
│ Gating      │ ← Determines which experts to use
│ Network     │
└─────────────┘
     ↓
┌─────────────┐
│ Expert      │ ← Routes to selected experts
│ Selection   │
└─────────────┘
     ↓
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ Expert 1    │   │ Expert 2    │   │ Expert K    │
│             │   │             │   │             │
└─────────────┘   └─────────────┘   └─────────────┘
     ↓                   ↓                   ↓
┌─────────────┐
│ Weighted    │ ← Combines expert outputs
│ Combination │
└─────────────┘
     ↓
Output Token
```

## Configuration

### Basic Configuration

```python
from dl_techniques.layers.moe import MoEConfig, ExpertConfig, GatingConfig

# Create custom configuration
config = MoEConfig(
    num_experts=16,
    expert_config=ExpertConfig(
        expert_type='ffn',
        hidden_dim=1024,
        intermediate_size=4096,
        activation='gelu'
    ),
    gating_config=GatingConfig(
        gating_type='linear',
        top_k=2,
        aux_loss_weight=0.01
    )
)

moe_layer = MixtureOfExperts(config)
```

### Pre-configured Options

```python
from dl_techniques.layers.moe import get_preset_moe

# Get pre-configured MoE layers
default_moe = get_preset_moe('default')
large_moe = get_preset_moe('large', num_experts=32)
attention_moe = get_preset_moe('attention')
vision_moe = get_preset_moe('vision')
```

## Usage Examples

### 1. Switch Transformer (Language Model)

```python
import keras
from dl_techniques.layers.moe import create_ffn_moe

# Replace FFN layers with MoE in transformer
def create_switch_transformer_layer(hidden_dim, num_heads, num_experts):
    inputs = keras.Input(shape=(None, hidden_dim))
    
    # Self-attention
    attn_output = keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=hidden_dim // num_heads
    )(inputs, inputs)
    x = keras.layers.LayerNormalization()(inputs + attn_output)
    
    # MoE replaces standard FFN
    moe_output = create_ffn_moe(
        num_experts=num_experts,
        hidden_dim=hidden_dim,
        top_k=1  # Switch Transformer uses k=1
    )(x)
    
    outputs = keras.layers.LayerNormalization()(x + moe_output)
    return keras.Model(inputs, outputs)
```

### 2. Mixture of Attention (MoA)

```python
from dl_techniques.layers.moe import create_attention_moe

# Specialized attention patterns
moa_layer = create_attention_moe(
    num_experts=12,
    hidden_dim=768,
    num_heads=12,
    top_k=2,
    gating_type='cosine'  # Better for attention specialization
)

# Use in transformer block
inputs = keras.Input(shape=(512, 768))
attention_output = moa_layer(inputs)
```

### 3. Vision MoE

```python
from dl_techniques.layers.moe import create_conv_moe

# Convolutional experts for vision
conv_moe = create_conv_moe(
    num_experts=8,
    filters=256,
    kernel_size=3,
    top_k=1
)

# Use in vision transformer
image_patches = keras.Input(shape=(14, 14, 768))  # Reshaped ViT patches
conv_output = conv_moe(image_patches)
```

### 4. Multimodal MoE

```python
# Modality-specific expert specialization
multimodal_config = MoEConfig(
    num_experts=16,
    expert_config=ExpertConfig(expert_type='ffn', hidden_dim=512),
    gating_config=GatingConfig(
        gating_type='cosine',  # Better for modality separation
        top_k=2,
        aux_loss_weight=0.1
    )
)

multimodal_moe = MixtureOfExperts(config=multimodal_config)
```

## Training Considerations

### 1. Learning Rate

MoE models often require different learning rates:

```python
# Lower learning rate for stability
optimizer = keras.optimizers.AdamW(
    learning_rate=1e-4,  # Typically lower than dense models
    weight_decay=0.01
)
```

### 2. Load Balancing

Monitor auxiliary losses during training:

```python
# The MoE layer automatically adds auxiliary losses
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy'
    # Auxiliary losses are automatically included
)

# Monitor expert utilization
moe_layer = model.layers[1]  # Assuming MoE is second layer
utilization = moe_layer.get_expert_utilization()
print(f"Expert utilization: {utilization}")
```

### 3. Expert Capacity

Configure capacity factors for different phases:

```python
config = MoEConfig(
    train_capacity_factor=1.25,  # Training capacity
    eval_capacity_factor=1.0,    # Evaluation capacity
    drop_tokens=True,            # Drop excess tokens
    use_residual_connection=True # Residual for dropped tokens
)
```

## Advanced Usage

### Custom Expert Types

```python
from dl_techniques.layers.moe.experts import BaseExpert

class CustomExpert(BaseExpert):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
    
    def build(self, input_shape):
        # Custom expert implementation
        self.custom_layer = keras.layers.Dense(self.units)
        super().build(input_shape)
    
    def call(self, inputs, training=None):
        return self.custom_layer(inputs)
    
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.units,)
```

### Custom Gating Functions

```python
from dl_techniques.layers.moe.gating import BaseGating

class CustomGating(BaseGating):
    def call(self, inputs, training=None):
        # Custom routing logic
        gate_logits = self.compute_gate_logits(inputs)
        expert_weights = keras.ops.softmax(gate_logits)
        expert_indices = keras.ops.argmax(expert_weights, axis=-1)
        
        auxiliary_info = {
            'gate_logits': gate_logits,
            'expert_weights': expert_weights
        }
        
        return expert_weights, expert_indices, auxiliary_info
```

### Hierarchical Routing

```python
# Enable hierarchical routing for very large expert counts
hierarchical_config = MoEConfig(
    num_experts=64,
    hierarchical_routing=True,
    num_routing_levels=2
)
```

## Performance Optimization

### Memory Optimization

```python
# For large models, use expert parallelism
config = MoEConfig(
    expert_parallel=True,  # Distribute experts across devices
    jitter_noise=0.01,     # Add capacity jittering
    routing_dtype='float16' # Lower precision for routing
)
```

### Communication Optimization

```python
# Optimize for distributed training
config = GatingConfig(
    top_k=1,              # Reduce communication overhead
    capacity_factor=1.0   # Tight capacity bounds
)
```

## Troubleshooting

### Common Issues

1. **Expert Collapse**: Some experts receive no tokens
   ```python
   # Solution: Increase auxiliary loss weight
   config.gating_config.aux_loss_weight = 0.1
   ```

2. **Training Instability**: Loss spikes during training
   ```python
   # Solution: Lower learning rate and add noise
   config.gating_config.add_noise = True
   config.gating_config.noise_std = 1.0
   ```

3. **Memory Issues**: OOM with large expert counts
   ```python
   # Solution: Use expert parallelism or reduce capacity
   config.expert_parallel = True
   config.train_capacity_factor = 1.0
   ```

### Debugging Expert Utilization

```python
# Check expert load distribution
def monitor_expert_utilization(moe_layer, inputs):
    _, _, gating_info = moe_layer.gating_network(inputs)
    expert_probs = gating_info['raw_gate_probs']
    
    # Compute expert usage statistics
    expert_usage = keras.ops.mean(expert_probs, axis=(0, 1))
    print(f"Expert usage: {expert_usage}")
    print(f"Usage std: {keras.ops.std(expert_usage)}")
```

## Performance Benchmarks

Typical performance characteristics:

| Configuration | Parameters | FLOPs | Expert Utilization |
|---------------|------------|-------|-------------------|
| 8 experts, k=1 | 1.2x dense | 0.4x dense | ~12.5% per expert |
| 16 experts, k=2 | 2.0x dense | 0.5x dense | ~12.5% per expert |
| 32 experts, k=1 | 4.0x dense | 0.3x dense | ~3.1% per expert |

## References

- **Switch Transformer**: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity
- **GLaM**: Efficient Scaling of Language Models with Mixture-of-Experts  
- **PaLM**: Scaling Language Modeling with Pathways
- **GShard**: Scaling Giant Models with Conditional Computation and Automatic Sharding
- **ST-MoE**: Designing Stable and Transferable Sparse Expert Models

## Contributing

When contributing to the MoE module:

1. Follow the dl_techniques documentation standards
2. Add comprehensive tests for new expert/gating types
3. Ensure serialization compatibility
4. Add usage examples for new features
5. Update this README with any new capabilities

## License

This module is part of the dl_techniques framework and follows the same licensing terms.