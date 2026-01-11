# Deep Learning Techniques - Optimization Module Guide

A comprehensive guide to using the optimization module for configuring optimizers, learning rate schedules, deep supervision, and advanced inference techniques in your deep learning projects.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Optimizer Builder](#optimizer-builder)
4. [Muon Optimizer](#muon-optimizer)
5. [Learning Rate Schedule Builder](#learning-rate-schedule-builder)
6. [Deep Supervision Schedule Builder](#deep-supervision-schedule-builder)
7. [SLED Logits Processor](#sled-logits-processor)
8. [Complete Integration Examples](#complete-integration-examples)
9. [Configuration Reference](#configuration-reference)
10. [Best Practices](#best-practices)
11. [Troubleshooting](#troubleshooting)
12. [Advanced Usage](#advanced-usage)

## Overview

The optimization module provides comprehensive components for configuring training optimization and inference:

- **Optimizer Builder**: Creates and configures Keras optimizers (Adam, AdamW, RMSprop, Adadelta) with gradient clipping support
- **Muon Optimizer**: MomentUm Orthogonalized by Newton-schulz optimizer for faster convergence on Transformers and ConvNets
- **Learning Rate Schedule Builder**: Creates learning rate schedules with automatic warmup periods
- **Deep Supervision Schedule Builder**: Creates weight schedules for multi-scale deep supervision training
- **SLED Logits Processor**: Self Logits Evolution Decoding for improving factuality in LLMs

### Key Features

- **Configuration-driven**: All components use dictionary-based configuration
- **Sensible defaults**: Fallback to proven default values from research
- **Flattened structure**: Simple, intuitive configuration format
- **Gradient clipping**: Built-in support for gradient clipping methods
- **Warmup periods**: Automatic learning rate warmup for training stability
- **Deep supervision**: Multiple scheduling strategies for multi-scale training
- **Backend-agnostic**: SLED and core components work across TensorFlow, PyTorch, and JAX backends

## Quick Start

### Basic Setup

```python
from dl_techniques.optimization import (
    optimizer_builder,
    learning_rate_schedule_builder,
    deep_supervision_schedule_builder
)

# 1. Configure learning rate schedule with warmup
lr_config = {
    "type": "cosine_decay",
    "warmup_steps": 1000,
    "warmup_start_lr": 1e-8,
    "learning_rate": 0.001,
    "decay_steps": 10000,
    "alpha": 0.0001
}

# 2. Configure optimizer
optimizer_config = {
    "type": "adam",
    "beta_1": 0.9,
    "beta_2": 0.999,
    "gradient_clipping_by_norm": 1.0
}

# 3. Build components
lr_schedule = learning_rate_schedule_builder(lr_config)
optimizer = optimizer_builder(optimizer_config, lr_schedule)

# 4. Use with your model
model.compile(optimizer=optimizer, loss='categorical_crossentropy')
```

### Training with Deep Supervision

```python
# Configure deep supervision weights
ds_config = {
    "type": "linear_low_to_high",
    "config": {}
}

# Build scheduler for 5 output scales
ds_scheduler = deep_supervision_schedule_builder(ds_config, 5)

# Use during training
def train_step(batch_data, epoch_progress):
    # Get current weights for this training progress
    supervision_weights = ds_scheduler(epoch_progress)  # [0.0, 1.0]
    
    # Apply weights to multiple outputs
    total_loss = 0
    for i, (output, weight) in enumerate(zip(model_outputs, supervision_weights)):
        scale_loss = loss_function(targets[i], output)
        total_loss += weight * scale_loss
    
    return total_loss
```

## Optimizer Builder

The optimizer builder creates configured Keras optimizers with gradient clipping support.

### Supported Optimizers

- **Adam**: Adaptive moment estimation with bias correction
- **AdamW**: Adam with decoupled weight decay (better for transformers)
- **RMSprop**: Root mean square propagation (good for RNNs)
- **Adadelta**: Adaptive learning rate method

### Basic Configuration

```python
# Minimal configuration (uses defaults)
config = {
    "type": "adam"
}
optimizer = optimizer_builder(config, learning_rate=0.001)

# Full configuration
config = {
    "type": "adam",
    "beta_1": 0.9,
    "beta_2": 0.999,
    "epsilon": 1e-7,
    "amsgrad": False,
    "gradient_clipping_by_value": 0.5,
    "gradient_clipping_by_norm_local": 1.0,
    "gradient_clipping_by_norm": 1.0
}
optimizer = optimizer_builder(config, lr_schedule)
```

### Optimizer-Specific Parameters

#### Adam/AdamW
```python
adam_config = {
    "type": "adam",  # or "adamw"
    "beta_1": 0.9,          # First moment decay rate
    "beta_2": 0.999,        # Second moment decay rate
    "epsilon": 1e-7,        # Numerical stability constant
    "amsgrad": False        # Use AMSGrad variant
}
```

#### RMSprop
```python
rmsprop_config = {
    "type": "rmsprop",
    "rho": 0.9,             # Decay factor for moving average
    "momentum": 0.0,        # Momentum factor
    "epsilon": 1e-7,        # Numerical stability constant
    "centered": False       # Use centered RMSprop
}
```

#### Adadelta
```python
adadelta_config = {
    "type": "adadelta",
    "rho": 0.9,             # Decay rate for gradient accumulation
    "epsilon": 1e-7         # Numerical stability constant
}
```

### Gradient Clipping Options

```python
config = {
    "type": "adam",
    # Clip gradients by absolute value (clips to [-0.5, 0.5])
    "gradient_clipping_by_value": 0.5,
    
    # Clip gradients by L2 norm per variable
    "gradient_clipping_by_norm_local": 1.0,
    
    # Clip gradients by global L2 norm
    "gradient_clipping_by_norm": 1.0
}
```

## Muon Optimizer

The Muon (MomentUm Orthogonalized by Newton-schulz) optimizer is a novel optimization algorithm designed for faster convergence on Transformers and ConvNets.

### Key Characteristics

1. **Hybrid Optimization**: Muon optimizes hidden linear transformation weights (rank >= 2) while an integrated auxiliary AdamW handles embeddings, normalization, biases, and classification heads.

2. **Newton-Schulz Orthogonalization**: Projects momentum updates onto the manifold of orthogonal matrices, enabling significantly larger learning rates (e.g., 0.02 vs 0.0003 for AdamW).

3. **Hardware Efficient**: Uses standard matrix multiplications, making it efficient on GPUs/TPUs and stable in lower precision (bfloat16).

### Performance Achievements

- ~1.35x training speedup for GPT-2 scale models compared to AdamW
- State-of-the-art training speed records for CIFAR-10
- Proven effectiveness at large batch sizes
- Adopted by frontier labs for large-scale LLM pre-training

### Basic Usage

```python
from dl_techniques.optimization.muon_optimizer import Muon

# Create Muon optimizer
optimizer = Muon(
    learning_rate=0.02,           # Muon LR (higher than typical AdamW)
    momentum=0.95,                # Momentum factor
    nesterov=True,                # Use Nesterov momentum
    ns_steps=5,                   # Newton-Schulz iterations
    adam_learning_rate=1e-3,      # AdamW LR for auxiliary params
    adam_beta_1=0.9,
    adam_beta_2=0.999,
    weight_decay=0.0
)

# Use with your model
model.compile(optimizer=optimizer, loss='categorical_crossentropy')
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 0.02 | Learning rate for Muon-optimized parameters |
| `momentum` | 0.95 | Momentum factor for Muon |
| `nesterov` | True | Whether to use Nesterov momentum |
| `ns_steps` | 5 | Number of Newton-Schulz iterations |
| `adam_learning_rate` | 1e-3 | Learning rate for AdamW auxiliary optimizer |
| `adam_beta_1` | 0.9 | First moment decay rate (Adam) |
| `adam_beta_2` | 0.999 | Second moment decay rate (Adam) |
| `adam_epsilon` | 1e-7 | Numerical stability constant (Adam) |
| `weight_decay` | 0.0 | Weight decay coefficient (decoupled) |
| `exclude_embedding_names` | ["embedding", "token_emb", "embed"] | Substrings to identify embedding layers |

### Automatic Parameter Routing

Muon automatically routes parameters to the appropriate optimizer:

- **Muon**: Weight matrices with rank >= 2 (excluding embeddings)
- **AdamW**: Biases, normalization parameters, embeddings, and 1D parameters

```python
# Parameters routed to Muon:
# - Dense layer kernels
# - Conv2D kernels
# - Attention projection weights

# Parameters routed to AdamW:
# - All biases
# - LayerNorm/BatchNorm parameters
# - Embedding tables
# - Final classification head
```

### Advanced Configuration

```python
# Transformer-optimized Muon configuration
optimizer = Muon(
    learning_rate=0.02,
    momentum=0.95,
    nesterov=True,
    ns_steps=5,
    adam_learning_rate=3e-4,      # Lower for embeddings
    adam_beta_2=0.95,             # Lower for transformers
    weight_decay=0.01,            # Apply weight decay
    exclude_embedding_names=[     # Custom embedding patterns
        "embedding", 
        "token_emb", 
        "embed",
        "position"
    ]
)
```

## Learning Rate Schedule Builder

Creates learning rate schedules with automatic warmup periods for training stability.

### Supported Schedules

- **Cosine Decay**: Smooth cosine-based decay
- **Exponential Decay**: Exponential learning rate reduction
- **Cosine Decay with Restarts**: Cosine decay with periodic restarts

### Flattened Configuration Format

All parameters are specified at the top level - no nested dictionaries needed!

```python
# Simple cosine decay with warmup
config = {
    "type": "cosine_decay",
    "warmup_steps": 1000,
    "warmup_start_lr": 1e-8,
    "learning_rate": 0.001,
    "decay_steps": 10000,
    "alpha": 0.0001
}
lr_schedule = learning_rate_schedule_builder(config)
```

### Schedule Types

#### 1. Cosine Decay
Smoothly decreases learning rate following a cosine curve.

```python
config = {
    "type": "cosine_decay",
    "learning_rate": 0.001,      # Initial learning rate
    "decay_steps": 10000,        # Steps to decay over
    "alpha": 0.0001,             # Minimum LR as fraction of initial (optional)
    "warmup_steps": 1000,        # Warmup period (optional)
    "warmup_start_lr": 1e-8      # Starting warmup LR (optional)
}
```

#### 2. Exponential Decay
Multiplicatively decreases learning rate at regular intervals.

```python
config = {
    "type": "exponential_decay",
    "learning_rate": 0.001,      # Initial learning rate
    "decay_steps": 1000,         # Steps between decay applications
    "decay_rate": 0.9,           # Multiplicative decay factor
    "warmup_steps": 500,         # Warmup period (optional)
    "warmup_start_lr": 1e-8      # Starting warmup LR (optional)
}
```

#### 3. Cosine Decay with Restarts
Cosine decay with periodic restarts to escape local minima.

```python
config = {
    "type": "cosine_decay_restarts",
    "learning_rate": 0.001,      # Initial learning rate
    "decay_steps": 5000,         # Steps in first decay period
    "t_mul": 2.0,                # Factor to multiply period after restart (optional)
    "m_mul": 0.9,                # Factor to multiply LR after restart (optional)
    "alpha": 0.001,              # Minimum LR as fraction (optional)
    "warmup_steps": 1000,        # Warmup period (optional)
    "warmup_start_lr": 1e-8      # Starting warmup LR (optional)
}
```

### Warmup Behavior

All schedules automatically include a linear warmup period:

- **Purpose**: Prevents training instability in early epochs
- **Behavior**: Linear increase from `warmup_start_lr` to target learning rate
- **Duration**: Specified by `warmup_steps` parameter
- **Default**: No warmup (`warmup_steps=0`) if not specified
- **Post-warmup**: Primary schedule starts from step 0 after warmup completes

```python
# During warmup (step < warmup_steps):
#   lr = warmup_start_lr + (target_lr - warmup_start_lr) * (step / warmup_steps)
# After warmup (step >= warmup_steps):
#   lr = primary_schedule(step - warmup_steps)
```

## Deep Supervision Schedule Builder

Creates weight schedules for training models with multiple output scales (e.g., U-Net architectures).

### Available Schedule Types

- **constant_equal**: Equal weights for all outputs
- **constant_low_to_high**: Fixed weights favoring higher resolution
- **constant_high_to_low**: Fixed weights favoring deeper layers
- **linear_low_to_high**: Gradual linear transition from deep to shallow focus
- **non_linear_low_to_high**: Quadratic transition for smoother focus shift
- **custom_sigmoid_low_to_high**: Sigmoid-based transition with custom parameters
- **scale_by_scale_low_to_high**: Progressive activation of outputs
- **cosine_annealing**: Oscillating weights with overall trend
- **curriculum**: Progressive activation based on training progress
- **step_wise**: Two-phase training with hard cutoff

### Basic Usage

```python
# Configure schedule
config = {
    "type": "linear_low_to_high",
    "config": {}  # Schedule-specific parameters (if any)
}

# Build scheduler for 5 output scales
num_outputs = 5
scheduler = deep_supervision_schedule_builder(config, num_outputs)

# Use during training
training_progress = 0.3  # 30% through training
weights = scheduler(training_progress)  # Returns array of 5 weights summing to 1.0
```

### Schedule Examples

#### 1. Linear Transition
Gradually shifts focus from deep (low-res) to shallow (high-res) outputs.

```python
config = {
    "type": "linear_low_to_high",
    "config": {}
}
```

#### 2. Custom Sigmoid Transition
Sigmoid-based transition with configurable parameters.

```python
config = {
    "type": "custom_sigmoid_low_to_high",
    "config": {
        "k": 10.0,                    # Sigmoid steepness
        "x0": 0.5,                    # Sigmoid midpoint
        "transition_point": 0.25      # When transition begins
    }
}
```

#### 3. Cosine Annealing
Oscillating weights with overall trend toward shallow outputs.

```python
config = {
    "type": "cosine_annealing",
    "config": {
        "frequency": 3.0,             # Number of cycles during training
        "final_ratio": 0.5            # Ratio between final and initial weights
    }
}
```

#### 4. Curriculum Learning
Progressive activation of outputs during training.

```python
config = {
    "type": "curriculum",
    "config": {
        "max_active_outputs": 3,      # Maximum simultaneously active outputs
        "activation_strategy": "linear"  # or "exp" for exponential
    }
}
```

#### 5. Step-Wise Transition
Two-phase training with linear transition then hard cutoff to final output.

```python
config = {
    "type": "step_wise",
    "config": {
        "threshold": 0.5              # Progress point for hard cutoff
    }
}
```

### Weight Order Inversion

By default, output 0 is the final inference output (highest resolution). Use `invert_order=True` if your architecture uses the opposite convention:

```python
scheduler = deep_supervision_schedule_builder(config, num_outputs, invert_order=True)
```

## SLED Logits Processor

Self Logits Evolution Decoding (SLED) is an inference-time framework that enhances factual accuracy of LLMs by leveraging "latent knowledge" from earlier layers.

### Key Features

- **No Fine-tuning Required**: Works at inference time only
- **Backend-Agnostic**: Fully portable across TensorFlow, PyTorch, and JAX
- **Top-k Optimization**: Efficient computation focused on top-k tokens

### Basic Usage

```python
from dl_techniques.optimization.sled_supervision import sled_builder

# Configure SLED processor
config = {
    "type": "sled_v1",
    "config": {
        "evolution_rate": 0.5,        # Alpha: magnitude of logit update
        "evolution_scale": 10,        # k: number of top tokens to consider
        "temperature": 1.0,           # Tau: softmax temperature
        "use_tau_in_update": True,    # Divide update by temperature
        "inactive_logit_value": -1e9  # Value for non-top-k tokens
    }
}

# Build processor
sled_processor = sled_builder(config)

# Use during generation
# all_logits_for_step: List of logits from each layer [batch_size, vocab_size]
evolved_logits = sled_processor(all_logits_for_step)
next_token = keras.ops.argmax(evolved_logits, axis=-1)
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `evolution_rate` | 0.5 | Alpha parameter controlling update magnitude |
| `evolution_scale` | 10 | Number of top-k tokens for evolution |
| `temperature` | 1.0 | Softmax temperature (must be > 0) |
| `use_tau_in_update` | True | Divide update term by temperature |
| `inactive_logit_value` | -1e9 | Logit value for non-top-k tokens |

### Algorithm Overview

SLED operates in three phases:

1. **Phase 1 - Estimate**: Compute gradient-based alignment scores between early and final layer logits
2. **Phase 2 - Ensemble**: Aggregate scores across all early layers to form latent knowledge estimate
3. **Phase 3 - Evolve**: Apply correction to final layer logits based on latent knowledge

## Complete Integration Examples

### Example 1: Basic Training Setup

```python
import keras
from dl_techniques.optimization import (
    optimizer_builder,
    learning_rate_schedule_builder
)

# Configure learning rate schedule
lr_config = {
    "type": "cosine_decay",
    "warmup_steps": 1000,
    "warmup_start_lr": 1e-8,
    "learning_rate": 0.001,
    "decay_steps": 10000,
    "alpha": 0.0001
}

# Configure optimizer
opt_config = {
    "type": "adamw",
    "beta_1": 0.9,
    "beta_2": 0.999,
    "gradient_clipping_by_norm": 1.0
}

# Build components
lr_schedule = learning_rate_schedule_builder(lr_config)
optimizer = optimizer_builder(opt_config, lr_schedule)

# Create and compile model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(x_train, y_train, epochs=50, batch_size=32)
```

### Example 2: Transformer Training with Muon

```python
import keras
from dl_techniques.optimization.muon_optimizer import Muon

# Create Muon optimizer for transformers
optimizer = Muon(
    learning_rate=0.02,
    momentum=0.95,
    nesterov=True,
    ns_steps=5,
    adam_learning_rate=3e-4,
    adam_beta_2=0.95,
    weight_decay=0.01,
    exclude_embedding_names=["embedding", "token_emb", "position"]
)

# Create transformer model
transformer_model = create_transformer_model()

# Compile with Muon
transformer_model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy'
)

# Train - Muon automatically routes parameters
transformer_model.fit(train_dataset, epochs=100)
```

### Example 3: U-Net with Deep Supervision

```python
import keras
import tensorflow as tf
import numpy as np
from dl_techniques.optimization import (
    optimizer_builder,
    learning_rate_schedule_builder,
    deep_supervision_schedule_builder
)

def create_unet_with_deep_supervision():
    """Create U-Net model with multiple output scales."""
    # ... model creation code ...
    # Returns model with 5 outputs at different scales
    pass

# Configure all components
lr_config = {
    "type": "cosine_decay_restarts",
    "warmup_steps": 2000,
    "learning_rate": 0.001,
    "decay_steps": 8000,
    "t_mul": 2.0,
    "m_mul": 0.8
}

opt_config = {
    "type": "adam",
    "gradient_clipping_by_norm": 0.5
}

ds_config = {
    "type": "linear_low_to_high",
    "config": {}
}

# Build components
lr_schedule = learning_rate_schedule_builder(lr_config)
optimizer = optimizer_builder(opt_config, lr_schedule)
ds_scheduler = deep_supervision_schedule_builder(ds_config, 5)

# Create model
model = create_unet_with_deep_supervision()

# Custom training loop with deep supervision
@tf.function
def train_step(x_batch, y_batch, epoch_progress):
    # Get current supervision weights
    supervision_weights = ds_scheduler(epoch_progress)
    
    with tf.GradientTape() as tape:
        # Forward pass - model returns 5 outputs
        outputs = model(x_batch, training=True)
        
        # Compute weighted loss
        total_loss = 0
        for i, (output, weight) in enumerate(zip(outputs, supervision_weights)):
            target = tf.image.resize(y_batch, output.shape[1:3])
            scale_loss = keras.losses.binary_crossentropy(target, output)
            total_loss += weight * tf.reduce_mean(scale_loss)
    
    # Backward pass
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return total_loss

# Training loop
epochs = 100
for epoch in range(epochs):
    epoch_progress = epoch / epochs
    for step, (x_batch, y_batch) in enumerate(train_dataset):
        loss = train_step(x_batch, y_batch, epoch_progress)
```

### Example 4: LLM Generation with SLED

```python
import keras
from dl_techniques.optimization.sled_supervision import sled_builder

# Configure SLED
sled_config = {
    "type": "sled_v1",
    "config": {
        "evolution_rate": 0.5,
        "evolution_scale": 10,
        "temperature": 1.0
    }
}

sled_processor = sled_builder(sled_config)

def generate_with_sled(model, input_ids, max_length=100):
    """Generate text using SLED-enhanced decoding."""
    generated = input_ids
    
    for _ in range(max_length):
        # Get logits from all layers (model must expose intermediate logits)
        all_layer_logits = model.get_all_layer_logits(generated)
        
        # Apply SLED evolution
        evolved_logits = sled_processor(all_layer_logits)
        
        # Sample next token
        next_token = keras.ops.argmax(evolved_logits[:, -1, :], axis=-1)
        generated = keras.ops.concatenate([generated, next_token[:, None]], axis=1)
        
        if next_token == eos_token_id:
            break
    
    return generated
```

## Configuration Reference

### Default Values

The module uses research-backed default values from the constants module:

#### Warmup Defaults
- `DEFAULT_WARMUP_STEPS = 0`
- `DEFAULT_WARMUP_START_LR = 1e-8`

#### Optimizer Defaults
- **Adam**: `beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False`
- **AdamW**: Same as Adam
- **RMSprop**: `rho=0.9, momentum=0.0, epsilon=1e-7, centered=False`
- **Adadelta**: `rho=0.9, epsilon=1e-7`

#### Muon Defaults
- `learning_rate=0.02`
- `momentum=0.95`
- `nesterov=True`
- `ns_steps=5`
- `adam_learning_rate=1e-3`
- `adam_beta_1=0.9, adam_beta_2=0.999, adam_epsilon=1e-7`
- `weight_decay=0.0`

#### Schedule Defaults
- **Cosine Decay**: `alpha=0.0001`
- **Cosine Restarts**: `t_mul=2.0, m_mul=0.9, alpha=0.001`

#### SLED Defaults
- `evolution_rate=0.5`
- `evolution_scale=10`
- `temperature=1.0`
- `use_tau_in_update=True`
- `inactive_logit_value=-1e9`

### Required vs Optional Parameters

#### Learning Rate Schedules

**All Schedules Require:**
- `type`: Schedule type string
- `learning_rate`: Initial learning rate

**Schedule-Specific Required:**
- `decay_steps`: All schedules
- `decay_rate`: Exponential decay only

**Always Optional:**
- `warmup_steps`: Defaults to 0 (no warmup)
- `warmup_start_lr`: Defaults to 1e-8
- `alpha`: Schedule-specific minimum LR ratio
- `t_mul`, `m_mul`: Cosine restarts only

#### Optimizers

**Required:**
- `type`: Optimizer type string

**Optional (all have defaults):**
- Optimizer-specific hyperparameters
- Gradient clipping parameters

## Best Practices

### Learning Rate Schedules

1. **Use Warmup**: Always include warmup for training stability
   ```python
   "warmup_steps": max(1000, total_steps // 20)  # 5% of training
   ```

2. **Choose Appropriate Schedules**:
   - **Cosine Decay**: General purpose, smooth decay
   - **Cosine Restarts**: When training plateaus, helps escape local minima
   - **Exponential**: When you need precise control over decay timing

3. **Scale Warmup with Model Size**:
   ```python
   # Larger models need longer warmup
   warmup_steps = min(10000, max(1000, model_params // 1000))
   ```

### Optimizers

1. **Choose Based on Architecture**:
   - **Muon**: Transformers, large ConvNets (fastest convergence)
   - **AdamW**: Transformers, large models (most stable)
   - **Adam**: General purpose, CNNs
   - **RMSprop**: RNNs, unstable gradients
   - **Adadelta**: When learning rate is hard to tune

2. **Muon Learning Rates**:
   ```python
   # Muon uses much higher LR than AdamW
   muon_lr = 0.02       # Good starting point
   adamw_lr = 0.0003    # Typical AdamW LR
   ```

3. **Gradient Clipping Guidelines**:
   ```python
   # Conservative clipping for stability
   "gradient_clipping_by_norm": 1.0  # Good default
   
   # Aggressive clipping for RNNs
   "gradient_clipping_by_norm": 0.5  # Prevent exploding gradients
   ```

### Deep Supervision

1. **Choose Schedule Based on Architecture**:
   - **linear_low_to_high**: Standard U-Net, gradual transition
   - **curriculum**: Complex models, progressive learning
   - **step_wise**: Two-phase training (features then refinement)
   - **constant_equal**: When all scales are equally important

2. **Monitor Weight Distribution**:
   ```python
   # Log weights during training
   if step % 100 == 0:
       weights = ds_scheduler(epoch_progress)
       logger.info(f"DS weights: {weights}")
   ```

### SLED

1. **Tune Evolution Rate**:
   ```python
   # Higher alpha = stronger correction
   "evolution_rate": 0.3  # Conservative
   "evolution_rate": 0.7  # Aggressive
   ```

2. **Adjust Top-k for Vocabulary Size**:
   ```python
   # Larger k for larger vocabularies
   "evolution_scale": 10   # Small vocab
   "evolution_scale": 50   # Large vocab (50k+)
   ```

## Troubleshooting

### Common Issues

#### 1. Training Instability
```python
# Symptoms: Loss spikes, NaN values
# Solutions:
config = {
    "type": "adam",
    "gradient_clipping_by_norm": 0.5,  # Reduce clipping threshold
    "warmup_steps": 2000,              # Increase warmup
    "warmup_start_lr": 1e-9,           # Lower starting LR
    "learning_rate": 0.0001            # Reduce peak LR
}
```

#### 2. Slow Convergence
```python
# Symptoms: Very slow loss decrease
# Solutions:
config = {
    "type": "cosine_decay_restarts",
    "learning_rate": 0.003,            # Increase peak LR
    "warmup_steps": 500,               # Reduce warmup
    "t_mul": 2.0,                      # Add restarts
    "decay_steps": 5000
}

# Or try Muon for faster convergence
optimizer = Muon(learning_rate=0.02)
```

#### 3. Muon Not Improving Over AdamW
```python
# Check parameter routing
for var in model.trainable_variables:
    uses_muon = optimizer._should_use_muon(var)
    print(f"{var.name}: {'Muon' if uses_muon else 'AdamW'}")

# Ensure weight matrices are rank >= 2
# Add custom embedding patterns if needed
optimizer = Muon(
    exclude_embedding_names=["embedding", "your_custom_pattern"]
)
```

#### 4. Deep Supervision Not Working
```python
# Check weight distribution
weights = ds_scheduler(0.5)  # Mid-training
print(f"Weights sum: {np.sum(weights)}")  # Should be 1.0
print(f"Weights: {weights}")

# If weights are too extreme, try:
config = {
    "type": "custom_sigmoid_low_to_high",
    "config": {
        "k": 5.0,                      # Reduce steepness
        "transition_point": 0.1        # Start transition earlier
    }
}
```

#### 5. SLED Returning Original Logits
```python
# Check for zero denominator warning in logs
# This means all layer contrasts were misaligned

# Try adjusting parameters:
config = {
    "type": "sled_v1",
    "config": {
        "evolution_scale": 20,         # Increase top-k
        "temperature": 0.8             # Lower temperature
    }
}
```

### Error Messages

#### "Missing required parameters"
```python
# Ensure all required parameters are provided
config = {
    "type": "cosine_decay",
    "learning_rate": 0.001,  # Required
    "decay_steps": 10000,    # Required
}
```

#### "Unknown optimizer/schedule type"
```python
# Check spelling and supported types
supported_optimizers = ["adam", "adamw", "rmsprop", "adadelta"]
supported_schedules = ["cosine_decay", "exponential_decay", "cosine_decay_restarts"]
supported_ds_schedules = [
    "constant_equal", "constant_low_to_high", "constant_high_to_low",
    "linear_low_to_high", "non_linear_low_to_high", "custom_sigmoid_low_to_high",
    "scale_by_scale_low_to_high", "cosine_annealing", "curriculum", "step_wise"
]
```

## Advanced Usage

### Custom Learning Rate Curves

Combine multiple schedules for complex learning rate curves:

```python
# Create custom multi-phase training
phase1_config = {
    "type": "exponential_decay",
    "warmup_steps": 1000,
    "learning_rate": 0.001,
    "decay_steps": 5000,
    "decay_rate": 0.8
}

phase2_config = {
    "type": "cosine_decay",
    "warmup_steps": 0,  # No warmup for second phase
    "learning_rate": 0.0005,
    "decay_steps": 10000,
    "alpha": 0.0001
}
```

### Muon with Learning Rate Schedules

```python
from dl_techniques.optimization import learning_rate_schedule_builder
from dl_techniques.optimization.muon_optimizer import Muon

# Create schedule for Muon's main learning rate
lr_config = {
    "type": "cosine_decay",
    "warmup_steps": 1000,
    "learning_rate": 0.02,  # Higher LR for Muon
    "decay_steps": 50000,
    "alpha": 0.001
}
lr_schedule = learning_rate_schedule_builder(lr_config)

# Use schedule with Muon
optimizer = Muon(
    learning_rate=lr_schedule,
    adam_learning_rate=1e-3  # AdamW uses fixed rate
)
```

### Dynamic Deep Supervision

Adapt supervision weights based on validation performance:

```python
class AdaptiveSupervisionCallback(keras.callbacks.Callback):
    def __init__(self, ds_scheduler, base_config, num_outputs):
        self.ds_scheduler = ds_scheduler
        self.base_config = base_config
        self.num_outputs = num_outputs
        self.best_val_loss = float('inf')
        
    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss', 0)
        
        if val_loss >= self.best_val_loss:
            # Switch to more aggressive schedule if plateauing
            new_config = {"type": "step_wise", "config": {"threshold": 0.3}}
            self.ds_scheduler = deep_supervision_schedule_builder(
                new_config, self.num_outputs
            )
        else:
            self.best_val_loss = val_loss
```

### Multi-GPU Training Considerations

```python
import tensorflow as tf

# Adjust learning rate for multi-GPU training
strategy = tf.distribute.MirroredStrategy()
num_gpus = strategy.num_replicas_in_sync

config = {
    "type": "cosine_decay",
    "learning_rate": 0.001 * num_gpus,  # Scale LR with batch size
    "warmup_steps": 1000 * num_gpus,    # Scale warmup proportionally
    "decay_steps": 10000
}

# For Muon, scale both learning rates
with strategy.scope():
    optimizer = Muon(
        learning_rate=0.02 * num_gpus,
        adam_learning_rate=1e-3 * num_gpus
    )
```

### Hyperparameter Sweeps

```python
def create_sweep_configs():
    """Generate configurations for hyperparameter sweeps."""
    sweep_configs = []
    
    # Standard optimizer sweep
    for lr in [0.0001, 0.0003, 0.001]:
        for opt_type in ["adam", "adamw"]:
            for clip_norm in [0.5, 1.0, 2.0]:
                sweep_configs.append({
                    "optimizer": {
                        "type": opt_type,
                        "gradient_clipping_by_norm": clip_norm
                    },
                    "lr_schedule": {
                        "type": "cosine_decay",
                        "learning_rate": lr,
                        "warmup_steps": 1000,
                        "decay_steps": 10000
                    }
                })
    
    # Muon sweep
    for muon_lr in [0.01, 0.02, 0.05]:
        for adam_lr in [1e-4, 3e-4, 1e-3]:
            sweep_configs.append({
                "optimizer": "muon",
                "muon_config": {
                    "learning_rate": muon_lr,
                    "adam_learning_rate": adam_lr,
                    "momentum": 0.95
                }
            })
    
    return sweep_configs
```