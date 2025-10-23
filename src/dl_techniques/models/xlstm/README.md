# xLSTM: Extended Long Short-Term Memory

A production-ready, fully-featured implementation of the **xLSTM** architecture in **Keras 3**, based on the paper ["xLSTM: Extended Long Short-Term Memory"](https://arxiv.org/abs/2405.04517) by Beck et al. (2024).

This implementation follows the `dl_techniques` framework standards and modern Keras 3 best practices, providing a modular, well-documented, and fully serializable codebase that works seamlessly across TensorFlow, PyTorch, and JAX backends.

---

## Table of Contents

1. [What is xLSTM?](#1-what-is-xlstm)
2. [Key Innovations](#2-key-innovations)
3. [About This Implementation](#3-about-this-implementation)
4. [Quick Start](#4-quick-start)
5. [Architecture Components](#5-architecture-components)
6. [Configuration Options](#6-configuration-options)
7. [Usage Examples](#7-usage-examples)
8. [Serialization](#8-serialization)
9. [Performance Tips](#9-performance-tips)
10. [Testing](#10-testing)
11. [Architecture Details](#11-architecture-details)
12. [Requirements](#12-requirements)
13. [Citation](#13-citation)

---

## 1. What is xLSTM?

Long Short-Term Memory (LSTM) networks were foundational to many early successes in deep learning, particularly in sequence modeling. However, they were largely superseded by Transformer-based models due to limitations in storage capacity and parallelizability.

**xLSTM** revisits the core ideas of LSTMs and enhances them with modern techniques to create a new architecture that is competitive with state-of-the-art Transformers and State Space Models (SSMs). It introduces two novel LSTM variants that are then combined to form a powerful and scalable sequence model capable of handling long-range dependencies and large-scale datasets.

### Why xLSTM?

The xLSTM architecture addresses fundamental limitations of classical LSTMs:

- **Limited Storage Capacity**: Traditional LSTMs store information in scalar vectors, limiting their memory capacity
- **Parallelization Constraints**: Recurrent nature prevents efficient parallelization during training
- **Scaling Issues**: Difficult to scale to very long sequences or large model sizes

By introducing exponential gating, matrix memory structures, and hybrid architectures, xLSTM achieves:

- ✅ **Enhanced Memory Capacity**: Matrix-based memory for storing richer representations
- ✅ **Efficient Parallelization**: mLSTM blocks can process sequences in parallel like Transformers
- ✅ **Competitive Performance**: Matches or exceeds Transformers and SSMs on various benchmarks
- ✅ **Scalability**: Trains efficiently on modern hardware (GPUs/TPUs)

---

## 2. Key Innovations

The xLSTM architecture is built on two new types of memory structures that work together to create a powerful hybrid model:

### 2.1 sLSTM (Scalar LSTM)

The **sLSTM** introduces two main modifications to the classic LSTM cell:

#### 1. **Exponential Gating**
Instead of sigmoid activation, the input gate uses an **exponential function**:
```
i_t = exp(W_i @ x_t + R_i @ h_{t-1} + b_i)
```

This allows for more flexible and multiplicative updates to the memory, enabling the model to make more dramatic changes to stored information when needed.

#### 2. **Normalizer State**
A new state `n_t` is added to stabilize the memory updates introduced by exponential gating:
```
n_t = f_t * n_{t-1} + i_t
h_t = o_t * (c_t / n_t)
```

This normalization prevents numerical instability and allows the model to maintain stable memory dynamics even with exponential gating.

**Key Capabilities**:
- ⚡ **Powerful Memory Revision**: Exceptional ability to revise and update stored information
- 🎯 **State Tracking**: Outstanding performance on tasks requiring long-term state tracking
- 🔄 **Recurrent Processing**: Processes sequences step-by-step, maintaining explicit recurrence
- 🧮 **Stabilization**: Numerical stability through the normalizer state and stabilization techniques

### 2.2 mLSTM (Matrix LSTM)

The **mLSTM** fundamentally redesigns the LSTM memory structure to overcome storage and parallelization limitations:

#### 1. **Matrix Memory**
The cell state is no longer a scalar vector but a full **matrix** `C_t`:
```
C_t ∈ R^(key_dim × value_dim)
```

This dramatically increases the model's storage capacity, allowing it to store rich key-value associations similar to attention mechanisms.

#### 2. **Covariance Update Rule**
Uses a covariance-based rule for memory updates:
```
C_t = f_t ⊙ C_{t-1} + i_t ⊙ (v_t ⊗ k_t^T)
```

This outer-product update creates associations between keys and values, enabling content-based memory retrieval.

#### 3. **Full Parallelizability**
The mLSTM **abandons hidden-to-hidden recurrence**, allowing it to process entire sequences in parallel during training, much like a Transformer. This makes it highly efficient on modern hardware (GPUs/TPUs).

**Key Capabilities**:
- 🚀 **High Storage Capacity**: Matrix memory stores complex associations
- ⚡ **Parallel Training**: Fully parallelizable for efficient GPU utilization
- 🎯 **Content-Based Retrieval**: Keys and values enable attention-like memory access
- 📊 **Multi-Head Architecture**: Supports multiple attention heads for diverse representations

### 2.3 The Hybrid xLSTM Architecture

The final **xLSTM model** combines both components in a **hybrid architecture** by stacking residual blocks of `sLSTM` and `mLSTM`:

```
Input → Embedding
   ↓
[mLSTM Blocks] × N  (fast, parallel, high-capacity)
   ↓
[sLSTM Blocks] × M  (recurrent, state-tracking)
   ↓
Final Normalization → Output
```

This hybrid approach achieves a unique balance:

- **Scalability and Efficiency** from parallel `mLSTM` blocks
- **Expressiveness and State Tracking** from recurrent `sLSTM` blocks
- **Competitive Performance** with Transformers and SSMs
- **Flexible Architecture** through configurable block ratios

The block composition is controlled by the `mlstm_ratio` parameter, allowing you to tune the architecture for your specific use case:
- Higher mLSTM ratio → More parallelization, better for long sequences
- Higher sLSTM ratio → More recurrence, better for complex state tracking

---

## 3. About This Implementation

This implementation provides a complete, production-ready xLSTM codebase that integrates seamlessly with the `dl_techniques` framework while following all Keras 3 best practices.

### 3.1 Features

#### **Modular Components**
The code is organized into logical, reusable layers:

- **`sLSTMCell`**: Core recurrent cell with exponential gating
- **`sLSTMLayer`**: Wrapped RNN layer for sequence processing
- **`mLSTMCell`**: Matrix memory cell with parallel processing
- **`mLSTMLayer`**: Wrapped RNN layer for matrix LSTM
- **`sLSTMBlock`**: Residual block with sLSTM (post-normalization architecture)
- **`mLSTMBlock`**: Residual block with mLSTM (pre-up-projection architecture)
- **`xLSTM`**: Complete model that stacks blocks to build the full architecture

#### **Framework Integration**
- ✅ **Normalization Factory**: Uses `create_normalization_layer()` for all normalization
  - Supports: layer_norm, rms_norm, batch_norm, band_rms, adaptive_band_rms, dynamic_tanh, etc.
- ✅ **FFN Factory**: Uses `create_ffn_layer()` for feed-forward networks
  - Supports: mlp, swiglu, geglu, glu, differential, residual, swin_mlp
- ✅ **Standard Blocks**: Follows project patterns for consistent architecture

#### **Keras 3 Best Practices**
- ✅ **Full Serialization**: Every layer implements `get_config()` with `@keras.saving.register_keras_serializable(package="xLSTM")`
- ✅ **Correct Build Logic**: Separates layer creation (`__init__`) from weight creation (`build`)
- ✅ **RNN Cell Pattern**: Proper Keras RNN Cell/Layer infrastructure
- ✅ **Backend Agnostic**: Built with `keras.ops`, runs on TensorFlow, PyTorch, or JAX
- ✅ **Type Hints**: Complete Python 3.11 type annotations
- ✅ **Documentation**: Comprehensive Sphinx-style docstrings with examples

#### **Production Quality**
- ✅ **Input Validation**: Clear error messages for invalid configurations
- ✅ **Regularization Support**: Full support for kernel, recurrent, and bias regularizers
- ✅ **Custom Initializers**: Configurable initializers for all weight matrices
- ✅ **Masking Support**: Built-in sequence masking capabilities
- ✅ **State Management**: Proper state handling for recurrent layers
- ✅ **Comprehensive Testing**: 18 test cases covering all functionality

---

## 4. Quick Start

### Basic Usage

```python
import keras
from xlstm_refactored import xLSTM

# Define model parameters
VOCAB_SIZE = 10000
EMBED_DIM = 512
NUM_LAYERS = 12
SEQUENCE_LENGTH = 256

# Create xLSTM model
model = xLSTM(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    num_layers=NUM_LAYERS,
    mlstm_ratio=0.5,  # 50% mLSTM, 50% sLSTM
)

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)

# Save the model
model.save('xlstm_model.keras')

# Load the model
loaded_model = keras.models.load_model('xlstm_model.keras')
```

### Modern Configuration

```python
# Use modern components from dl_techniques
model = xLSTM(
    vocab_size=50000,
    embed_dim=768,
    num_layers=24,
    mlstm_ratio=0.5,
    mlstm_num_heads=12,
    mlstm_expansion_factor=2,
    slstm_forget_gate='exp',  # Exponential gating
    ffn_type='swiglu',  # SwiGLU FFN (as in LLaMa)
    ffn_expansion_factor=4,
    normalization_type='rms_norm',  # RMS normalization
    normalization_kwargs={'epsilon': 1e-6},
    dropout_rate=0.1,
    embedding_dropout_rate=0.1,
    kernel_regularizer=keras.regularizers.L2(1e-4),
)
```

---

## 5. Architecture Components

### 5.1 sLSTM Cell and Layer

#### **sLSTMCell**

Core recurrent cell with exponential gating and normalizer state:

```python
from xlstm_refactored import sLSTMCell

# Create cell
cell = sLSTMCell(
    units=128,
    forget_gate_activation='exp',  # 'sigmoid' or 'exp'
    kernel_initializer='glorot_uniform',
    recurrent_initializer='orthogonal',
)

# Single timestep forward pass
batch_size = 32
input_dim = 64
x_t = keras.random.normal((batch_size, input_dim))

# Initialize states
state = cell.get_initial_state(batch_size=batch_size)

# Process timestep
output, new_state = cell(x_t, state)
```

#### **sLSTMLayer**

Wrapped RNN layer for full sequence processing:

```python
from xlstm_refactored import sLSTMLayer

# Basic usage
layer = sLSTMLayer(units=128, return_sequences=True)
outputs = layer(inputs)  # inputs: (batch, seq_len, input_dim)

# With exponential forget gate
layer = sLSTMLayer(
    units=128,
    forget_gate_activation='exp',
    return_sequences=True,
    return_state=False,
)

# With state return
layer = sLSTMLayer(
    units=128,
    return_sequences=False,
    return_state=True,
)
output, h_state, c_state, n_state, m_state = layer(inputs)
```

**Key Features**:
- Exponential gating for improved memory dynamics
- Normalizer state (n_t) for stable memory updates
- Stabilization technique to prevent numerical overflow
- Compatible with standard Keras RNN API
- Support for masking, stateful mode, and bidirectional processing

### 5.2 mLSTM Cell and Layer

#### **mLSTMCell**

Matrix memory cell with covariance-style updates:

```python
from xlstm_refactored import mLSTMCell

# Create cell
cell = mLSTMCell(
    units=256,
    num_heads=4,
    key_dim=64,  # Optional, defaults to units // num_heads
    value_dim=64,  # Optional, defaults to units // num_heads
)

# Use with RNN
rnn = keras.layers.RNN(cell, return_sequences=True)
outputs = rnn(inputs)
```

#### **mLSTMLayer**

Wrapped RNN layer for matrix LSTM:

```python
from xlstm_refactored import mLSTMLayer

# Basic usage
layer = mLSTMLayer(
    units=256,
    num_heads=4,
    return_sequences=True,
)

# Custom key/value dimensions
layer = mLSTMLayer(
    units=256,
    num_heads=8,
    key_dim=64,
    value_dim=64,
)
```

**Key Features**:
- Matrix-valued memory C_t for enhanced capacity
- Multi-head architecture for parallel processing
- Fully parallelizable during training
- Covariance-style update rule
- Content-based memory retrieval

### 5.3 sLSTM Block

Residual block with post-normalization architecture (Transformer-style):

```python
from xlstm_refactored import sLSTMBlock

# Standard block
block = sLSTMBlock(
    units=256,
    ffn_type='swiglu',
    normalization_type='layer_norm',
)

# Custom configuration
block = sLSTMBlock(
    units=256,
    ffn_type='geglu',
    ffn_expansion_factor=4,
    normalization_type='rms_norm',
    normalization_kwargs={'epsilon': 1e-6},
    forget_gate_activation='exp',
    dropout_rate=0.1,
)
```

**Architecture Flow**:
```
Input (residual)
   ↓
sLSTMLayer
   ↓
Normalization
   ↓
Feed-Forward Network
   ↓
Add(residual) → Output
```

### 5.4 mLSTM Block

Residual block with pre-up-projection architecture (SSM-style):

```python
from xlstm_refactored import mLSTMBlock

# Standard block
block = mLSTMBlock(
    units=256,
    expansion_factor=2,
    num_heads=4,
)

# Custom configuration
block = mLSTMBlock(
    units=256,
    expansion_factor=3,
    num_heads=8,
    conv_kernel_size=7,
    normalization_type='rms_norm',
)
```

**Architecture Flow**:
```
Input (residual)
   ↓
Up-Projection (Dense)
   ↓
Depthwise Conv1D (causal)
   ↓
Activation (swish)
   ↓
mLSTMLayer
   ↓
Normalization
   ↓
Down-Projection (Dense)
   ↓
Add(residual) → Output
```

### 5.5 Full xLSTM Model

Complete model with stacked blocks:

```python
from xlstm_refactored import xLSTM

model = xLSTM(
    vocab_size=50000,
    embed_dim=512,
    num_layers=12,
    mlstm_ratio=0.5,  # 6 mLSTM blocks, 6 sLSTM blocks
    mlstm_num_heads=8,
    ffn_type='swiglu',
    normalization_type='rms_norm',
)
```

**Architecture**:
```
Tokens
   ↓
Embedding
   ↓
Optional Dropout
   ↓
[mLSTM Blocks] × (num_layers * mlstm_ratio)
   ↓
[sLSTM Blocks] × (num_layers * (1 - mlstm_ratio))
   ↓
Final Normalization
   ↓
Output Head (Dense)
```

---

## 6. Configuration Options

### 6.1 Normalization Types

Supported via `dl_techniques.layers.norms` factory:

| Type | Description | Use Case |
|------|-------------|----------|
| `'layer_norm'` | Standard layer normalization | Default, general purpose |
| `'rms_norm'` | Root Mean Square normalization | Faster than LayerNorm |
| `'batch_norm'` | Batch normalization | CNN-style architectures |
| `'band_rms'` | Band-constrained RMS | Stability-critical applications |
| `'adaptive_band_rms'` | Adaptive Band RMS | Dynamic stability control |
| `'dynamic_tanh'` | Dynamic Tanh normalization | Normalization-free transformers |

**Example**:
```python
block = sLSTMBlock(
    units=256,
    normalization_type='rms_norm',
    normalization_kwargs={'epsilon': 1e-6, 'use_scale': True}
)
```

### 6.2 FFN Types

Supported via `dl_techniques.layers.ffn` factory:

| Type | Description | Use Case |
|------|-------------|----------|
| `'mlp'` | Standard multi-layer perceptron | General purpose |
| `'swiglu'` | SwiGLU with gating | Modern LLMs (LLaMa, Qwen) |
| `'geglu'` | GELU-based gated linear unit | GELU-based architectures |
| `'glu'` | Standard gated linear unit | Improved gradient flow |
| `'differential'` | Dual-pathway processing | Enhanced feature processing |
| `'residual'` | Residual block with skip connections | Very deep networks |
| `'swin_mlp'` | Swin Transformer MLP | Vision models |

**Example**:
```python
block = sLSTMBlock(
    units=256,
    ffn_type='swiglu',
    ffn_expansion_factor=4,
    dropout_rate=0.1,
)
```

### 6.3 Regularization

Full support for regularizers:

```python
model = xLSTM(
    vocab_size=50000,
    embed_dim=512,
    num_layers=12,
    kernel_regularizer=keras.regularizers.L2(1e-4),
    recurrent_regularizer=keras.regularizers.L2(1e-4),
    bias_regularizer=None,
)
```

### 6.4 Initializers

Configurable initializers for all weight matrices:

```python
model = xLSTM(
    vocab_size=50000,
    embed_dim=512,
    num_layers=12,
    kernel_initializer='glorot_uniform',
    recurrent_initializer='orthogonal',
    bias_initializer='zeros',
)
```

---

## 7. Usage Examples

### 7.1 Language Modeling

```python
from xlstm_refactored import xLSTM
import keras

# Create language model
model = xLSTM(
    vocab_size=50000,
    embed_dim=768,
    num_layers=24,
    mlstm_ratio=0.5,
    mlstm_num_heads=12,
    mlstm_expansion_factor=2,
    slstm_forget_gate='exp',
    ffn_type='swiglu',
    ffn_expansion_factor=4,
    normalization_type='rms_norm',
    dropout_rate=0.1,
    embedding_dropout_rate=0.1,
)

# Compile with AdamW
model.compile(
    optimizer=keras.optimizers.AdamW(
        learning_rate=1e-4,
        weight_decay=0.01,
    ),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

# Train
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=100,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=5),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
        keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True),
    ],
)
```

### 7.2 Sequence Classification

```python
from xlstm_refactored import sLSTMLayer, mLSTMLayer
import keras

# Build classifier
inputs = keras.Input(shape=(None, 128))

# Stack xLSTM layers
x = mLSTMLayer(256, num_heads=4, return_sequences=True)(inputs)
x = sLSTMLayer(256, return_sequences=True)(x)
x = mLSTMLayer(256, num_heads=4, return_sequences=False)(x)

# Classification head
x = keras.layers.Dropout(0.3)(x)
outputs = keras.layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs, outputs)

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### 7.3 Time Series Prediction

```python
from xlstm_refactored import sLSTMBlock, mLSTMBlock
import keras

# Multi-scale temporal model
inputs = keras.Input(shape=(None, 64))

# Alternate between fast and slow processing
x = inputs
for i in range(6):
    if i % 2 == 0:
        # Fast: mLSTM for quick patterns
        x = mLSTMBlock(64, num_heads=4, name=f'mlstm_{i}')(x)
    else:
        # Slow: sLSTM for complex dependencies
        x = sLSTMBlock(64, name=f'slstm_{i}')(x)

# Prediction head
outputs = keras.layers.Dense(1)(x)

model = keras.Model(inputs, outputs)
```

### 7.4 Custom Hybrid Architecture

```python
from xlstm_refactored import sLSTMBlock, mLSTMBlock
import keras

# Front-end: Fast mLSTM processing
inputs = keras.Input(shape=(None, 256))
x = inputs

# Early layers: Fast parallel processing
for i in range(4):
    x = mLSTMBlock(
        256,
        expansion_factor=2,
        num_heads=8,
        name=f'frontend_mlstm_{i}'
    )(x)

# Late layers: Deep reasoning with sLSTM
for i in range(4):
    x = sLSTMBlock(
        256,
        ffn_type='swiglu',
        ffn_expansion_factor=4,
        forget_gate_activation='exp',
        name=f'backend_slstm_{i}'
    )(x)

# Output
outputs = keras.layers.Dense(vocab_size)(x)

model = keras.Model(inputs, outputs)
```

### 7.5 Complete Training Pipeline

```python
import keras
from xlstm_refactored import xLSTM

# 1. Create model
model = xLSTM(
    vocab_size=10000,
    embed_dim=512,
    num_layers=12,
    mlstm_ratio=0.5,
    ffn_type='swiglu',
    normalization_type='rms_norm',
)

# 2. Compile with mixed precision
keras.mixed_precision.set_global_policy('mixed_float16')

model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

# 3. Create datasets
train_dataset = create_dataset('train.txt', batch_size=32)
val_dataset = create_dataset('val.txt', batch_size=32)

# 4. Train with callbacks
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    callbacks=[
        keras.callbacks.TensorBoard(log_dir='./logs'),
        keras.callbacks.ModelCheckpoint(
            'checkpoints/model_{epoch:02d}.keras',
            save_freq='epoch'
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
    ],
)

# 5. Save final model
model.save('xlstm_final.keras')
```

---

## 8. Serialization

All components support full Keras 3 serialization:

### Save and Load

```python
# Save model
model.save('xlstm_model.keras')

# Load model
loaded_model = keras.models.load_model('xlstm_model.keras')

# Verify loaded model
predictions = loaded_model(test_inputs)
```

### Export Weights Only

```python
# Save weights
model.save_weights('xlstm_weights.weights.h5')

# Create new model with same architecture
new_model = xLSTM(vocab_size=50000, embed_dim=512, num_layers=12)

# Load weights
new_model.load_weights('xlstm_weights.weights.h5')
```

### Configuration Export

```python
# Get model configuration
config = model.get_config()

# Save to JSON
import json
with open('model_config.json', 'w') as f:
    json.dump(config, f, indent=2)

# Recreate model from config
from xlstm_refactored import xLSTM
new_model = xLSTM.from_config(config)
```

---

## 9. Performance Tips

### 9.1 Use mLSTM for Long Sequences

mLSTM is fully parallelizable during training:

```python
# Better for long sequences (training)
model = xLSTM(
    vocab_size=50000,
    embed_dim=512,
    num_layers=12,
    mlstm_ratio=0.67,  # More mLSTM = more parallelization
)
```

### 9.2 Use sLSTM for Inference

sLSTM maintains explicit recurrence for autoregressive generation:

```python
# Better for autoregressive generation (inference)
model = xLSTM(
    vocab_size=50000,
    embed_dim=512,
    num_layers=12,
    mlstm_ratio=0.33,  # More sLSTM = better recurrence
)
```

### 9.3 Balance Block Ratios

```python
# Balanced (recommended for most tasks)
mlstm_ratio=0.5  # 50-50 split

# Parallelization-heavy (long training sequences)
mlstm_ratio=0.67  # 67% mLSTM, 33% sLSTM

# Recurrence-heavy (complex state tracking)
mlstm_ratio=0.33  # 33% mLSTM, 67% sLSTM
```

### 9.4 Use RMS Normalization

RMSNorm is faster and more memory-efficient:

```python
model = xLSTM(
    vocab_size=50000,
    embed_dim=512,
    num_layers=12,
    normalization_type='rms_norm',  # Faster than layer_norm
)
```

### 9.5 Enable Mixed Precision

```python
# Enable mixed precision for faster training
keras.mixed_precision.set_global_policy('mixed_float16')

model = xLSTM(
    vocab_size=50000,
    embed_dim=512,
    num_layers=12,
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
)
```

### 9.6 Optimize Batch Size

```python
# Larger batches for mLSTM (parallel processing)
if mlstm_ratio > 0.5:
    batch_size = 64
else:
    batch_size = 32
```

---

## 10. Testing

Run the comprehensive test suite:

```bash
python test_xlstm.py
```

### Test Coverage

The test suite includes 18 comprehensive tests:

1. ✅ **Cell Tests**: Basic functionality and variants
2. ✅ **Layer Tests**: Sequence processing and state return
3. ✅ **Block Tests**: Residual connections and architectures
4. ✅ **Model Tests**: Full model instantiation and configurations
5. ✅ **Serialization Tests**: Save/load for all components
6. ✅ **Training Tests**: Compile, fit, and gradient flow
7. ✅ **Configuration Tests**: Different norm/FFN types
8. ✅ **Error Handling**: Input validation

**Expected Output**:
```
======================================================================
Running xLSTM Implementation Tests
======================================================================

Testing sLSTMCell basic functionality...
✓ sLSTMCell basic test passed

[... 17 more tests ...]

======================================================================
Test Results: 18 passed, 0 failed
======================================================================
```

---

## 11. Architecture Details

### 11.1 sLSTM Mathematical Formulation

At each timestep t:

**Gates**:
```
i_t = exp(W_i x_t + R_i h_{t-1} + b_i)        # Input gate (exponential)
f_t = σ(W_f x_t + R_f h_{t-1} + b_f)          # Forget gate (sigmoid or exp)
o_t = σ(W_o x_t + R_o h_{t-1} + b_o)          # Output gate
z_t = tanh(W_z x_t + R_z h_{t-1} + b_z)       # Cell input
```

**Stabilization** (Equations 15-17 in paper):
```
m_t = max(m_{t-1} + log(f_t), log(i_t))       # Stabilizer state
ĩ_t = exp(log(i_t) - m_t)                     # Stabilized input gate
f̃_t = exp(log(f_t) + m_{t-1} - m_t)          # Stabilized forget gate
```

**State Updates**:
```
c_t = f̃_t ⊙ c_{t-1} + ĩ_t ⊙ z_t              # Cell state
n_t = f̃_t ⊙ n_{t-1} + ĩ_t                    # Normalizer state
```

**Output**:
```
h_t = o_t ⊙ (c_t / (n_t + ε))                 # Hidden state
```

### 11.2 mLSTM Mathematical Formulation

**Projections**:
```
q_t = W_q x_t + R_q h_{t-1}                   # Query
k_t = W_k x_t + R_k h_{t-1}                   # Key
v_t = W_v x_t + R_v h_{t-1}                   # Value
```

**Gates**:
```
i_t = exp(W_i x_t + R_i h_{t-1})              # Input gate (exponential)
f_t = σ(W_f x_t + R_f h_{t-1})                # Forget gate
o_t = σ(W_o x_t + R_o h_{t-1})                # Output gate
```

**Matrix Memory Update**:
```
C_t = f_t ⊙ C_{t-1} + i_t ⊙ (v_t ⊗ k_t^T)     # Covariance update
```

**Normalizer Update**:
```
n_t = f_t ⊙ n_{t-1} + i_t ⊙ k_t               # Normalizer
```

**Output**:
```
h_t = o_t ⊙ (C_t q_t / (n_t^T q_t + ε))       # Retrieved hidden state
```

### 11.3 Comparison with Standard LSTM

| Feature | Standard LSTM | sLSTM | mLSTM |
|---------|--------------|-------|-------|
| Memory Type | Scalar vector | Scalar vector | Matrix |
| Gating | Standard (sigmoid) | Exponential | Exponential + Standard |
| Normalizer | No | Yes | Yes |
| Stabilization | No | Yes | Yes |
| Parallelizable | No | No | Yes (training) |
| Memory Capacity | Limited | Enhanced | High |
| Computational Cost | Low | Medium | Higher |
| Storage Revision | Limited | Excellent | Good |
| Long-Range Dependencies | Moderate | Good | Excellent |

---

## 12. Requirements

### Core Dependencies

```
python>=3.11
keras>=3.8.0
tensorflow>=2.18.0  # or torch, or jax
```

### Framework Integration

This implementation requires the `dl_techniques` framework for:
- Normalization factory (`dl_techniques.layers.norms`)
- FFN factory (`dl_techniques.layers.ffn`)

### Installation

```bash
# Install Keras and backend
pip install keras tensorflow

# Install dl_techniques (if available)
pip install dl_techniques

# Or use the standalone implementation
```

---

## 13. Citation

If you use xLSTM in your research, please cite the original paper:

```bibtex
@article{beck2024xlstm,
  title={xLSTM: Extended Long Short-Term Memory},
  author={Beck, Maximilian and P{\"o}ppel, Korbinian and Spanring, Markus and 
          Auer, Andreas and Prudnikova, Oleksandra and Kopp, Michael and 
          Klambauer, G{\"u}nter and Brandstetter, Johannes and Hochreiter, Sepp},
  journal={arXiv preprint arXiv:2405.04517},
  year={2024}
}
```

---

## License

This implementation is provided for research and educational purposes. See the LICENSE file for details.

---

## Acknowledgments

This implementation:
- Follows the architecture described in Beck et al. (2024)
- Uses the `dl_techniques` framework for normalization and FFN factories
- Adheres to Keras 3 best practices for custom layers and models
- Implements proper serialization according to modern Keras standards

---
