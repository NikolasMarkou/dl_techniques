# Deep Learning Techniques - Optimization Module Guide

A comprehensive guide to using the optimization module for configuring optimizers, learning rate schedules, and deep supervision in your deep learning projects.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Optimizer Builder](#optimizer-builder)
4. [Learning Rate Schedule Builder](#learning-rate-schedule-builder)
5. [Deep Supervision Schedule Builder](#deep-supervision-schedule-builder)
6. [Complete Integration Examples](#complete-integration-examples)
7. [Configuration Reference](#configuration-reference)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Usage](#advanced-usage)

## Overview

The optimization module provides three main components for configuring training optimization:

- **Optimizer Builder**: Creates and configures Keras optimizers (Adam, AdamW, RMSprop, Adadelta) with gradient clipping support
- **Learning Rate Schedule Builder**: Creates learning rate schedules with automatic warmup periods
- **Deep Supervision Schedule Builder**: Creates weight schedules for multi-scale deep supervision training

### Key Features

- **Configuration-driven**: All components use dictionary-based configuration
- **Sensible defaults**: Fallback to proven default values from research
- **Flattened structure**: Simple, intuitive configuration format
- **Gradient clipping**: Built-in support for gradient clipping methods
- **Warmup periods**: Automatic learning rate warmup for training stability
- **Deep supervision**: Multiple scheduling strategies for multi-scale training

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

```python
# During warmup: lr = warmup_start_lr + (target_lr - warmup_start_lr) * (step / warmup_steps)
# After warmup: lr = primary_schedule(step)
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
        "final_ratio": 0.8            # Ratio between final and initial weights
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

### Example 2: U-Net with Deep Supervision

```python
import keras
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
            # Each output corresponds to a different scale
            target = tf.image.resize(y_batch, output.shape[1:3])
            scale_loss = keras.losses.binary_crossentropy(target, output)
            total_loss += weight * tf.reduce_mean(scale_loss)
    
    # Backward pass
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return total_loss

# Training loop
epochs = 100
steps_per_epoch = len(train_dataset)

for epoch in range(epochs):
    epoch_progress = epoch / epochs
    
    for step, (x_batch, y_batch) in enumerate(train_dataset):
        loss = train_step(x_batch, y_batch, epoch_progress)
        
        if step % 100 == 0:
            weights = ds_scheduler(epoch_progress)
            print(f"Epoch {epoch}, Step {step}, Loss: {loss:.4f}")
            print(f"Supervision weights: {weights}")
```

### Example 3: Transformer Training with Advanced Schedules

```python
# Transformer-optimized configuration
lr_config = {
    "type": "cosine_decay_restarts",
    "warmup_steps": 4000,           # Longer warmup for transformers
    "warmup_start_lr": 1e-9,        # Very low starting LR
    "learning_rate": 0.0005,        # Conservative peak LR
    "decay_steps": 20000,
    "t_mul": 1.5,
    "m_mul": 0.95,
    "alpha": 0.01
}

opt_config = {
    "type": "adamw",                # AdamW better for transformers
    "beta_1": 0.9,
    "beta_2": 0.98,                 # Higher beta_2 for transformers
    "epsilon": 1e-9,
    "gradient_clipping_by_norm": 1.0
}

# Build and use
lr_schedule = learning_rate_schedule_builder(lr_config)
optimizer = optimizer_builder(opt_config, lr_schedule)

# Create transformer model and train...
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

#### Schedule Defaults
- **Cosine Decay**: `alpha=0.0001`
- **Cosine Restarts**: `t_mul=2.0, m_mul=0.9, alpha=0.001`

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
   - **AdamW**: Transformers, large models
   - **Adam**: General purpose, CNNs
   - **RMSprop**: RNNs, unstable gradients
   - **Adadelta**: When learning rate is hard to tune

2. **Gradient Clipping Guidelines**:
   ```python
   # Conservative clipping for stability
   "gradient_clipping_by_norm": 1.0  # Good default
   
   # Aggressive clipping for RNNs
   "gradient_clipping_by_norm": 0.5  # Prevent exploding gradients
   ```

3. **Hyperparameter Tuning**:
   ```python
   # Start with defaults, then tune
   # Lower beta_2 for noisy gradients
   "beta_2": 0.98 if transformer_model else 0.999
   ```

### Deep Supervision

1. **Choose Schedule Based on Architecture**:
   - **linear_low_to_high**: Standard U-Net, gradual transition
   - **curriculum**: Complex models, progressive learning
   - **constant_equal**: When all scales are equally important

2. **Monitor Weight Distribution**:
   ```python
   # Log weights during training
   if step % 100 == 0:
       weights = ds_scheduler(epoch_progress)
       logger.info(f"DS weights: {weights}")
   ```

3. **Adjust Based on Training Behavior**:
   - If early layers aren't learning: Use `constant_high_to_low`
   - If final output is poor: Use `constant_low_to_high`
   - For balanced learning: Use `linear_low_to_high`

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
```

#### 3. Deep Supervision Not Working
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

#### 4. Memory Issues
```python
# Large models with gradient clipping
config = {
    "type": "adamw",
    "gradient_clipping_by_value": 0.1,  # Use value clipping instead of norm
    # Remove global norm clipping to save memory
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
    # Optional parameters can be omitted
}
```

#### "Unknown optimizer/schedule type"
```python
# Check spelling and supported types
supported_optimizers = ["adam", "adamw", "rmsprop", "adadelta"]
supported_schedules = ["cosine_decay", "exponential_decay", "cosine_decay_restarts"]
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
    "learning_rate": 0.0005,  # Lower starting LR
    "decay_steps": 10000,
    "alpha": 0.0001
}

# Use different schedules for different training phases
```

### Dynamic Deep Supervision

Adapt supervision weights based on validation performance:

```python
class AdaptiveSupervisionCallback(keras.callbacks.Callback):
    def __init__(self, ds_scheduler, base_config):
        self.ds_scheduler = ds_scheduler
        self.base_config = base_config
        self.best_val_loss = float('inf')
        
    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss', 0)
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
        else:
            # Adapt schedule if validation loss plateaus
            # Switch to more aggressive schedule
            new_config = self.base_config.copy()
            new_config['type'] = 'curriculum'
            self.ds_scheduler = deep_supervision_schedule_builder(new_config, 5)
```

### Multi-GPU Training Considerations

```python
# Adjust learning rate for multi-GPU training
strategy = tf.distribute.MirroredStrategy()
num_gpus = strategy.num_replicas_in_sync

config = {
    "type": "cosine_decay",
    "learning_rate": 0.001 * num_gpus,  # Scale LR with batch size
    "warmup_steps": 1000 * num_gpus,    # Scale warmup proportionally
    "decay_steps": 10000
}
```

### Hyperparameter Sweeps

```python
def create_sweep_configs():
    """Generate configurations for hyperparameter sweeps."""
    base_config = {
        "type": "adam",
        "warmup_steps": 1000,
        "decay_steps": 10000
    }
    
    sweep_configs = []
    
    # Sweep learning rates
    for lr in [0.0001, 0.0003, 0.001, 0.003]:
        # Sweep optimizer types
        for opt_type in ["adam", "adamw"]:
            # Sweep gradient clipping
            for clip_norm in [0.5, 1.0, 2.0]:
                config = base_config.copy()
                config.update({
                    "learning_rate": lr,
                    "type": opt_type,
                    "gradient_clipping_by_norm": clip_norm
                })
                sweep_configs.append(config)
    
    return sweep_configs
```