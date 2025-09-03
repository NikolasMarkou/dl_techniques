# Mixture of Experts (MoE) Module

A production-ready Mixture of Experts implementation for the dl_techniques framework, providing sparse neural network architectures through conditional computation with FFN expert specialization.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Basic Usage](#basic-usage)
6. [Advanced Usage](#advanced-usage)
7. [Integration](#integration)
8. [API Reference](#api-reference)
9. [Performance Considerations](#performance-considerations)
10. [Troubleshooting](#troubleshooting)

## Overview

The MoE module implements sparse neural networks where each input is routed to a subset of expert networks, enabling:

- **Computational Efficiency**: Only selected experts are activated per input
- **Model Specialization**: Experts learn to handle specific input patterns
- **Scalable Architecture**: Add capacity without proportional compute increase
- **Load Balancing**: Auxiliary losses prevent expert collapse

### Key Components

- **Expert Networks**: FFN-based specialists using dl_techniques FFN factory
- **Gating Networks**: Routing mechanisms (linear, cosine similarity, SoftMoE)
- **Load Balancing**: Auxiliary and z-losses for uniform expert utilization
- **Capacity Management**: Token dropping and residual connections

## Architecture

### MoE Layer Structure

```
Input → Gating Network → Expert Selection → Expert Processing → Output
         ↓                ↓                  ↓
         Router           Top-K Selection    Weighted Combination
         Logits           Expert Indices     Final Output
```

### Expert Types

The module exclusively uses FFN experts, leveraging the dl_techniques FFN factory:

- **MLP**: Standard multi-layer perceptron (`type: "mlp"`)
- **SwiGLU**: Gated linear units with SiLU activation (`type: "swiglu"`)
- **GeGLU**: GELU-based gated linear units (`type: "geglu"`)
- **GLU**: Standard gated linear units (`type: "glu"`)
- **Differential**: Dual-pathway processing (`type: "differential"`)
- **Residual**: Skip connections for gradient flow (`type: "residual"`)
- **Swin MLP**: Vision-optimized MLP variant (`type: "swin_mlp"`)

### Gating Mechanisms

- **Linear Gating**: Standard learnable routing with optional noise
- **Cosine Gating**: Cosine similarity-based routing in hypersphere space
- **SoftMoE**: Soft routing with weighted token slots per expert

## Installation

The MoE module is part of dl_techniques and requires:

```python
import keras
from dl_techniques.layers.moe import MixtureOfExperts, MoEConfig, ExpertConfig, GatingConfig
```

### Dependencies

- Keras 3.8.0+
- TensorFlow 2.18.0 (backend)
- dl_techniques FFN module
- NumPy, typing

## Configuration

### Basic Configuration

```python
from dl_techniques.layers.moe import MoEConfig, ExpertConfig, GatingConfig

# Expert configuration using FFN factory
expert_config = ExpertConfig(
    ffn_config={
        "type": "swiglu",           # FFN type
        "d_model": 768,             # Model dimension
        "ffn_expansion_factor": 4   # Expansion ratio
    }
)

# Gating configuration
gating_config = GatingConfig(
    gating_type='linear',      # Routing mechanism
    top_k=2,                   # Experts per token
    aux_loss_weight=0.01       # Load balancing weight
)

# Complete MoE configuration
moe_config = MoEConfig(
    num_experts=8,
    expert_config=expert_config,
    gating_config=gating_config
)
```

### Configuration Classes

#### ExpertConfig

```python
@dataclass
class ExpertConfig:
    ffn_config: Dict[str, Any]                                    # FFN configuration dict
    use_bias: bool = True                                         # Bias in additional layers
    kernel_initializer: Union[str, Initializer] = 'glorot_uniform'
    bias_initializer: Union[str, Initializer] = 'zeros'
    kernel_regularizer: Optional[Regularizer] = None
    bias_regularizer: Optional[Regularizer] = None
```

#### GatingConfig

```python
@dataclass  
class GatingConfig:
    gating_type: Literal['linear', 'cosine', 'softmoe'] = 'linear'
    top_k: int = 1                    # Experts selected per token
    capacity_factor: float = 1.25     # Expert capacity multiplier
    add_noise: bool = True            # Exploration noise
    noise_std: float = 1.0            # Noise standard deviation
    temperature: float = 1.0          # Softmax temperature
    
    # Linear gating
    use_bias: bool = False
    
    # Cosine gating  
    embedding_dim: int = 256
    learnable_temperature: bool = True
    
    # SoftMoE
    num_slots: int = 4
    
    # Load balancing
    aux_loss_weight: float = 0.01     # Auxiliary loss weight
    z_loss_weight: float = 1e-3       # Router z-loss weight
```

#### MoEConfig

```python
@dataclass
class MoEConfig:
    num_experts: int = 8
    expert_config: ExpertConfig = field(default_factory=ExpertConfig)
    gating_config: GatingConfig = field(default_factory=GatingConfig)
    
    # System parameters
    jitter_noise: float = 0.01
    drop_tokens: bool = True
    use_residual_connection: bool = True
    
    # Training parameters  
    train_capacity_factor: Optional[float] = None
    eval_capacity_factor: Optional[float] = None
    routing_dtype: str = 'float32'
```

## Basic Usage

### Simple MoE Layer

```python
import keras
from dl_techniques.layers.moe import MixtureOfExperts, create_ffn_moe

# Method 1: Using convenience function
moe_layer = create_ffn_moe(
    num_experts=8,
    ffn_config={
        "type": "swiglu",
        "d_model": 768,
        "ffn_expansion_factor": 4
    },
    top_k=2,
    gating_type='linear'
)

# Method 2: Using configuration classes
from dl_techniques.layers.moe import MoEConfig, ExpertConfig, GatingConfig

config = MoEConfig(
    num_experts=8,
    expert_config=ExpertConfig(
        ffn_config={
            "type": "mlp",
            "hidden_dim": 2048, 
            "output_dim": 768,
            "activation": "gelu"
        }
    ),
    gating_config=GatingConfig(gating_type='linear', top_k=2)
)

moe_layer = MixtureOfExperts(config)
```

### Integration in Models

```python
import keras
from dl_techniques.layers.moe import create_ffn_moe

def create_transformer_with_moe():
    inputs = keras.Input(shape=(128, 768))
    
    # Standard transformer layers
    x = keras.layers.MultiHeadAttention(num_heads=12, key_dim=64)(inputs, inputs)
    x = keras.layers.LayerNormalization()(x + inputs)
    
    # Replace FFN with MoE
    residual = x
    moe_output = create_ffn_moe(
        num_experts=8,
        ffn_config={
            "type": "swiglu",
            "d_model": 768,
            "ffn_expansion_factor": 4
        },
        top_k=2
    )(x)
    x = keras.layers.LayerNormalization()(moe_output + residual)
    
    outputs = keras.layers.Dense(vocab_size)(x)
    return keras.Model(inputs, outputs)
```

## Advanced Usage

### Custom FFN Expert Types

```python
# GeGLU experts with custom parameters
config = MoEConfig(
    num_experts=16,
    expert_config=ExpertConfig(
        ffn_config={
            "type": "geglu",
            "hidden_dim": 3072,
            "output_dim": 768,
            "dropout_rate": 0.1
        }
    ),
    gating_config=GatingConfig(gating_type='cosine', top_k=1)
)

# Differential FFN experts
config = MoEConfig(
    num_experts=12,
    expert_config=ExpertConfig(
        ffn_config={
            "type": "differential",
            "hidden_dim": 1024,
            "output_dim": 768,
            "branch_activation": "relu",
            "combination_activation": "gelu"
        }
    )
)
```

### Advanced Gating Configurations

```python
# Cosine similarity gating
cosine_config = GatingConfig(
    gating_type='cosine',
    top_k=1,
    embedding_dim=256,
    temperature=0.1,
    learnable_temperature=True,
    aux_loss_weight=0.02
)

# SoftMoE gating (no hard routing)
softmoe_config = GatingConfig(
    gating_type='softmoe',
    num_slots=4,
    aux_loss_weight=0.01,
    z_loss_weight=1e-3
)
```

### Training Configuration

```python
from dl_techniques.layers.moe.integration import MoETrainingConfig, MoEOptimizerBuilder

# MoE-optimized training configuration
training_config = MoETrainingConfig(
    optimizer_type='adamw',
    base_learning_rate=1e-4,
    expert_learning_rate_multiplier=0.1,  # Lower LR for experts
    warmup_steps=2000,
    aux_loss_weight=0.01,
    weight_decay=0.01,
    gradient_clipping_norm=1.0
)

# Build MoE-optimized optimizer
builder = MoEOptimizerBuilder()
optimizer = builder.build_moe_optimizer(model, training_config)
```

### Load Balancing and Regularization

```python
config = MoEConfig(
    num_experts=16,
    expert_config=expert_config,
    gating_config=GatingConfig(
        gating_type='linear',
        top_k=2,
        capacity_factor=1.25,          # Expert capacity
        add_noise=True,                # Exploration noise
        noise_std=1.0,
        aux_loss_weight=0.01,          # Load balancing
        z_loss_weight=1e-3             # Entropy regularization
    ),
    # Token management
    drop_tokens=True,
    use_residual_connection=True,
    jitter_noise=0.01                  # Input noise for regularization
)
```

## Integration

### With dl_techniques Optimization

```python
from dl_techniques.optimization import optimizer_builder, learning_rate_schedule_builder

# Create learning rate schedule
lr_config = {
    "type": "cosine_decay",
    "warmup_steps": 1000,
    "learning_rate": 1e-4,
    "decay_steps": 10000
}

# Create optimizer
opt_config = {
    "type": "adamw",
    "gradient_clipping_by_norm": 1.0
}

lr_schedule = learning_rate_schedule_builder(lr_config)
optimizer = optimizer_builder(opt_config, lr_schedule)

# Compile model with MoE layers
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')
```

### With Model Analyzer

```python
from dl_techniques.analyzer import ModelAnalyzer, AnalysisConfig

# Analyze MoE model performance
config = AnalysisConfig(
    analyze_weights=True,
    analyze_training_dynamics=True,
    save_plots=True
)

analyzer = ModelAnalyzer(
    models={'moe_model': moe_model},
    config=config,
    output_dir='moe_analysis'
)
results = analyzer.analyze(data=test_data)

# Get expert utilization statistics
utilization = moe_layer.get_expert_utilization()
```

## API Reference

### MixtureOfExperts Layer

```python
class MixtureOfExperts(keras.layers.Layer):
    def __init__(self, config: MoEConfig, **kwargs)
    def call(self, inputs, training=None) -> keras.KerasTensor
    def compute_output_shape(self, input_shape) -> Tuple[Optional[int], ...]
    def get_expert_utilization(self) -> Dict[str, Any]
    def get_config(self) -> Dict[str, Any]
    def from_config(cls, config: Dict[str, Any]) -> 'MixtureOfExperts'
```

### Factory Functions

```python
def create_ffn_moe(
    num_experts: int,
    ffn_config: Dict[str, Any], 
    top_k: int = 1,
    gating_type: str = 'linear',
    aux_loss_weight: float = 0.01,
    **kwargs
) -> MixtureOfExperts
```

### Expert Networks

```python
class FFNExpert(BaseExpert):
    def __init__(self, ffn_config: Dict[str, Any], **kwargs)
    def call(self, inputs, training=None) -> keras.KerasTensor
    def compute_output_shape(self, input_shape) -> Tuple[Optional[int], ...]
```

### Gating Networks

```python
def create_gating(
    gating_type: str, 
    num_experts: int, 
    **kwargs
) -> BaseGating

class LinearGating(BaseGating):
    def __init__(self, num_experts, top_k=1, use_bias=False, add_noise=True, ...)

class CosineGating(BaseGating):
    def __init__(self, num_experts, embedding_dim=256, top_k=1, ...)

class SoftMoEGating(BaseGating): 
    def __init__(self, num_experts, num_slots=4, ...)
```

### Auxiliary Loss Functions

```python
def compute_auxiliary_loss(
    expert_weights: keras.KerasTensor,
    gate_probs: keras.KerasTensor, 
    num_experts: int,
    aux_loss_weight: float = 0.01
) -> keras.KerasTensor

def compute_z_loss(
    gate_logits: keras.KerasTensor,
    z_loss_weight: float = 1e-3
) -> keras.KerasTensor
```


## Training Best Practices

### Optimizer Configuration

```python
from dl_techniques.layers.moe.integration import MoETrainingConfig

# Recommended training configuration
training_config = MoETrainingConfig(
    optimizer_type='adamw',
    base_learning_rate=1e-4,
    expert_learning_rate_multiplier=0.1,    # Lower LR for experts
    gating_learning_rate_multiplier=1.0,    # Normal LR for gating
    warmup_steps=2000,
    decay_steps=50000,
    weight_decay=0.01,
    gradient_clipping_norm=1.0,
    aux_loss_weight=0.01
)
```

### Loss Monitoring

```python
# Custom training loop with MoE loss monitoring
@tf.function
def train_step(batch_x, batch_y):
    with tf.GradientTape() as tape:
        predictions = model(batch_x, training=True)
        
        # Main task loss
        task_loss = loss_fn(batch_y, predictions)
        
        # MoE auxiliary losses (automatically added)
        total_loss = task_loss + sum(model.losses)
    
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return {
        'task_loss': task_loss,
        'aux_losses': sum(model.losses),
        'total_loss': total_loss
    }
```

## Performance Considerations

### Memory Usage

- **Expert Capacity**: Balance between load balancing and memory usage
- **Top-K Selection**: Higher top_k increases computation but improves quality
- **Token Dropping**: Reduces memory but may hurt performance

### Computational Efficiency

```python
# Efficient configuration for large models
efficient_config = MoEConfig(
    num_experts=32,               # Many experts
    expert_config=ExpertConfig(
        ffn_config={
            "type": "swiglu",     # Efficient gated FFN
            "d_model": 512,       # Moderate size
            "ffn_expansion_factor": 2  # Lower expansion
        }
    ),
    gating_config=GatingConfig(
        gating_type='linear',     # Fastest gating
        top_k=1,                  # Minimal routing
        capacity_factor=1.0       # Tight capacity
    )
)
```

### Load Balancing Tuning

```python
# Monitor expert utilization
def monitor_expert_usage(moe_layer):
    stats = moe_layer.get_expert_utilization()
    print(f"Experts: {stats['num_experts']}")
    print(f"Routing: {stats['routing_type']}")
    print(f"Top-K: {stats['top_k']}")
    print(f"Capacity: {stats['expert_capacity_train']}")

# Adjust auxiliary loss weights based on utilization
# High aux_loss_weight (0.1+) for better load balancing
# Low aux_loss_weight (0.001) for minimal interference
```

## Model Serialization

### Save and Load

```python
# Save model with MoE layers
model.save('moe_model.keras')

# Load model (automatic registration)
loaded_model = keras.models.load_model('moe_model.keras')

# Verify expert configuration preserved
moe_layer = loaded_model.get_layer('mixture_of_experts')
utilization = moe_layer.get_expert_utilization()
print(f"Loaded model has {utilization['num_experts']} experts")
```

### Configuration Export

```python
# Export configuration for reproducibility
config_dict = moe_config.to_dict()

# Save configuration
import json
with open('moe_config.json', 'w') as f:
    json.dump(config_dict, f, indent=2)

# Load and recreate
with open('moe_config.json', 'r') as f:
    loaded_config_dict = json.load(f)
    
loaded_config = MoEConfig.from_dict(loaded_config_dict)
moe_layer = MixtureOfExperts(loaded_config)
```

## Troubleshooting

### Common Issues

#### Expert Collapse
**Symptoms**: Some experts never receive tokens, poor performance
```python
# Solution: Increase auxiliary loss weight
gating_config = GatingConfig(
    aux_loss_weight=0.1,      # Increase from 0.01
    z_loss_weight=1e-2        # Increase entropy regularization
)
```

#### Memory Issues
**Symptoms**: OOM errors during training
```python
# Solution: Reduce expert capacity or enable token dropping
config = MoEConfig(
    train_capacity_factor=1.0,    # Reduce from 1.25
    drop_tokens=True,
    use_residual_connection=True  # Handle dropped tokens
)
```

#### Training Instability
**Symptoms**: Loss spikes, gradient explosions
```python
# Solution: Lower learning rates and add gradient clipping
training_config = MoETrainingConfig(
    expert_learning_rate_multiplier=0.05,  # Much lower for experts
    gradient_clipping_norm=0.5,            # Tighter clipping
    jitter_noise=0.001                     # Less input noise
)
```

#### Poor Expert Utilization
**Symptoms**: Experts have very uneven usage
```python
# Solution: Tune capacity and noise parameters
gating_config = GatingConfig(
    capacity_factor=1.5,          # More capacity
    add_noise=True,              # Enable exploration
    noise_std=2.0,               # Increase exploration
    aux_loss_weight=0.05         # Stronger load balancing
)
```

### Debugging

#### Expert Utilization Analysis

```python
def analyze_expert_usage(model, dataset):
    """Analyze how experts are being used."""
    
    # Get MoE layers
    moe_layers = [layer for layer in model.layers 
                  if isinstance(layer, MixtureOfExperts)]
    
    for moe_layer in moe_layers:
        stats = moe_layer.get_expert_utilization()
        print(f"\nMoE Layer: {moe_layer.name}")
        print(f"Configuration: {stats}")
        
        # Check auxiliary losses
        if hasattr(moe_layer, '_auxiliary_losses'):
            aux_losses = moe_layer._auxiliary_losses
            print(f"Auxiliary losses: {len(aux_losses)}")
```

#### Validation

```python
def validate_moe_model(model, sample_input):
    """Validate MoE model functionality."""
    
    # Test forward pass
    output = model(sample_input)
    print(f"Forward pass successful: {output.shape}")
    
    # Test training mode
    with tf.GradientTape() as tape:
        training_output = model(sample_input, training=True)
        loss = keras.ops.mean(keras.ops.square(training_output))
    
    gradients = tape.gradient(loss, model.trainable_variables)
    print(f"Gradients computed: {len(gradients)} variables")
    
    # Test serialization
    try:
        model.save('test_moe.keras')
        loaded = keras.models.load_model('test_moe.keras')
        print("Serialization test passed")
    except Exception as e:
        print(f"Serialization failed: {e}")
```

### Error Messages

#### "FFN configuration validation failed"
- Check that `ffn_config` contains 'type' field
- Verify FFN parameters match the selected type requirements
- Use `validate_ffn_config()` from dl_techniques.layers.ffn

#### "Unknown gating type"
- Supported types: 'linear', 'cosine', 'softmoe'
- Check spelling and case sensitivity

#### "Expert capacity exceeded"
- Increase `capacity_factor` in gating configuration
- Enable `drop_tokens=True` and `use_residual_connection=True`
- Reduce `top_k` to decrease expert load

## Examples

### Complete Training Script

```python
import keras
from dl_techniques.layers.moe import create_ffn_moe
from dl_techniques.optimization import optimizer_builder, learning_rate_schedule_builder

# Create model with MoE
inputs = keras.Input(shape=(128, 768))
x = inputs

# Add MoE FFN layer
moe_layer = create_ffn_moe(
    num_experts=8,
    ffn_config={
        "type": "swiglu",
        "d_model": 768,
        "ffn_expansion_factor": 4
    },
    top_k=2,
    aux_loss_weight=0.01
)
x = moe_layer(x)

outputs = keras.layers.Dense(vocab_size)(x)
model = keras.Model(inputs, outputs)

# Configure optimizer for MoE
lr_config = {"type": "cosine_decay", "learning_rate": 1e-4, "decay_steps": 10000}
opt_config = {"type": "adamw", "gradient_clipping_by_norm": 1.0}

lr_schedule = learning_rate_schedule_builder(lr_config)
optimizer = optimizer_builder(opt_config, lr_schedule)

# Compile and train
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')
model.fit(train_data, epochs=10, validation_data=val_data)

# Monitor expert utilization
stats = moe_layer.get_expert_utilization()
print(f"Expert utilization: {stats}")
```



## References

- **Switch Transformer**: [Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)
- **GLaM**: [Efficient Scaling of Language Models with Mixture-of-Experts](https://arxiv.org/abs/2112.06905)  
- **SoftMoE**: [From Sparse to Soft Mixtures of Experts](https://arxiv.org/abs/2308.00951)
- **dl_techniques FFN Module**: See `layers/ffn/README.md` for FFN factory documentation