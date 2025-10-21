# Tiny Recursive Model (TRM)

This repository contains a Keras 3 implementation of the **Tiny Recursive Model (TRM)**, a highly parameter-efficient architecture for complex reasoning tasks. This implementation is a direct adaptation of the original PyTorch model from the paper "[Less is More: Recursive Reasoning with Tiny Networks](https://arxiv.org/abs/2510.04871)".

The core idea of TRM is to achieve powerful reasoning capabilities not by scaling up model size, but by recursively applying a small, shared neural network to refine its internal state and predictions over a variable number of steps. This approach is inspired by the principles of **Adaptive Computation Time (ACT)**, allowing the model to dynamically allocate more "thinking" time to more difficult problems.

This implementation prioritizes clarity, modularity, and adherence to Keras 3 best practices, making it easy to integrate, understand, and serialize within the TensorFlow ecosystem.

## References and Original Work

This work would not be possible without the foundational research and open-source contributions of the original authors. Please cite their work if you use this model.

-   **Paper**: Jolicoeur-Martineau, A. (2025). *Less is More: Recursive Reasoning with Tiny Networks*. [arXiv:2510.04871](https://arxiv.org/abs/2510.04871).
-   **Original PyTorch Repository**: [samsungsailmontreal/tinyrecursivemodels](https://github.com/samsungsailmontreal/tinyrecursivemodels)
-   **Inspiration**: Wang, G., et al. (2025). *Hierarchical Reasoning Model*. [arXiv:2506.21734](https://arxiv.org/abs/2506.21734).
-   **ACT Reference**: Graves, A. (2016). *Adaptive Computation Time for Recurrent Neural Networks*. [NeurIPS 2016](https://proceedings.neurips.cc/paper/2016/file/bf69145244511593c662c12aee4608c0-Paper.pdf).

## Core Concepts

The model is built on three fundamental principles:

### 1. Recursive Reasoning
Instead of a deep, feed-forward network, TRM uses a small, shallow network that is called repeatedly. At each step, it takes its own previous output (its latent state) as input, allowing it to progressively refine its "thought process." This makes the model extremely parameter-efficient.

### 2. Adaptive Computation Time (ACT)
The model learns *how many* recursive steps are needed for a given problem. At each step, a dedicated "halting head" (`q_head`) outputs probabilities for whether to stop or continue reasoning. This allows the model to save computation on easy examples and dedicate more resources to harder ones.

The halting mechanism supports two modes:
- **Simple Halting** (`no_act_continue=True`): Halts when `q_halt > 0`
- **Q-Learning Halting** (`no_act_continue=False`): Halts when `q_halt > q_continue`, using lookahead for Bellman updates

### 3. Hierarchical Latent States
The model maintains two levels of latent states:
- **`z_H` (high-level)**: Represents abstract reasoning state
- **`z_L` (low-level)**: Represents detailed processing state

These states are updated at each step through two reasoning modules (`H_level` and `L_level`), with `z_L` incorporating new input embeddings and `z_H` building upon `z_L`.

## Architectural Breakdown

The implementation in `model.py` is structured into three main Keras classes, promoting modularity and reusability.

### 1. `TRMReasoningModule` (`keras.layers.Layer`)

**Purpose**: A configurable stack of transformer layers forming the core computational unit.

**Architecture**:
```
Input → TransformerLayer_0 → TransformerLayer_1 → ... → TransformerLayer_N → Output
```

**Key Features**:
- Stacks `num_layers` instances of `TransformerLayer`
- Each layer shares the same architecture configuration
- Configurable attention type (default: `multi_head`)
- Configurable FFN type (default: `swiglu`)
- Configurable normalization (default: `rms_norm`, position: `post`)

**Function**: Takes a latent state and an "input injection" tensor, adds them together, and processes through the transformer stack.

### 2. `TRMInner` (`keras.layers.Layer`)

**Purpose**: Orchestrates one complete reasoning step, managing the hierarchical state updates.

**Architecture**:
```
Input Tokens
     ↓
Token Embedding
     ↓
[Puzzle Padding] + [Token Embeddings]
     ↓
L_level: z_L ← TRMReasoningModule(z_L, input_emb)
     ↓
H_level: z_H ← TRMReasoningModule(z_H, z_L)
     ↓
     ├→ LM Head → logits (predictions)
     └→ Q Head → (q_halt, q_continue) (halting probabilities)
```

**Key Components**:
- **Token Embedding**: Projects input token IDs to hidden dimension
- **Puzzle Embedding Padding**: Adds learnable prefix tokens (default: 16 positions)
- **L_level Module**: `TRMReasoningModule` with `l_layers` transformer layers
- **H_level Module**: `TRMReasoningModule` with `h_layers` transformer layers
- **LM Head**: Dense layer producing vocabulary logits
- **Q Head**: Dense layer producing halting decision logits
- **Learnable Initial States**: `H_init` and `L_init` weights for state reset

**State Flow**:
1. Low-level state `z_L` is updated by combining previous `z_L` with new input embeddings
2. High-level state `z_H` is updated by combining previous `z_H` with updated `z_L`
3. States are detached from gradient graph (`tf.stop_gradient`) for next iteration
4. Output logits and halting logits are generated from final `z_H`

### 3. `TinyRecursiveReasoningModel` (`keras.Model`)

**Purpose**: Main model class managing the outer ACT loop and persistent state.

**Key Responsibilities**:

#### State Management
The model maintains a `carry` dictionary containing:
- **`inner_carry`**: Dictionary with `z_H` and `z_L` latent states
- **`steps`**: Integer counter tracking steps per batch item
- **`halted`**: Boolean mask indicating which items have stopped
- **`current_data`**: Current input data for each item

#### Single Step Operation
The `call` method performs **one** step of reasoning:
1. **Reset**: Replaces `z_H` and `z_L` with learned initial states (`H_init`, `L_init`) for newly started sequences
2. **Update**: Passes states through `TRMInner` for computation
3. **Halt Decision**: Determines which sequences should halt based on:
   - Maximum step limit (`halt_max_steps`)
   - Learned halting signals (`q_halt`, `q_continue`)
   - Exploration probability during training (`halt_exploration_prob`)

#### Training vs Inference Behavior
- **Training**: Uses learned halting with exploration for better coverage
- **Inference**: Only halts at maximum step limit for consistent computation

#### Initial Carry Creation
The `initial_carry` method creates zero-initialized states with `halted=True`, triggering a state reset on the first step.

## Model Parameters

### Core Architecture Parameters
- **`vocab_size`** (int, required): Vocabulary size for token embeddings
- **`hidden_size`** (int, required): Hidden state dimensionality
- **`num_heads`** (int, required): Number of attention heads
- **`expansion`** (float, required): FFN expansion factor (intermediate_size = hidden_size × expansion)
- **`seq_len`** (int, required): Input sequence length (excluding puzzle embedding)

### Layer Configuration
- **`h_layers`** (int, default: 2): Number of transformer layers in H_level module
- **`l_layers`** (int, default: 2): Number of transformer layers in L_level module
- **`puzzle_emb_len`** (int, default: 16): Length of learnable puzzle embedding prefix

### Adaptive Computation Time (ACT) Parameters
- **`halt_max_steps`** (int, default: 10): Maximum reasoning steps allowed
- **`halt_exploration_prob`** (float, default: 0.1): Exploration probability for halting during training
- **`no_act_continue`** (bool, default: True): Use simple halting (True) vs Q-learning (False)

### Transformer Configuration
- **`rope_theta`** (float, default: 10000.0): RoPE theta parameter
- **`attention_type`** (str, default: 'multi_head'): Attention mechanism type
- **`ffn_type`** (str, default: 'swiglu'): Feed-forward network type
- **`normalization_type`** (str, default: 'rms_norm'): Normalization layer type
- **`normalization_position`** (str, default: 'post'): Normalization position ('pre' or 'post')
- **`dropout_rate`** (float, default: 0.0): General dropout rate
- **`attention_dropout_rate`** (float, default: 0.0): Attention-specific dropout rate

## Key Differences from Original Implementation

This implementation stays true to the logic of the original TRM but adapts it to the Keras/TensorFlow paradigm.

### Framework
- **Original**: PyTorch with JAX-style functional patterns
- **This Implementation**: Native Keras 3 on TensorFlow backend

### Architecture Modularity
- **Original**: Single-file implementation with nested functions
- **This Implementation**: Three distinct classes (`TRMReasoningModule`, `TRMInner`, `TinyRecursiveReasoningModel`) following Keras patterns

### Keras-Native Features
- **Serialization**: All layers registered with `@keras.saving.register_keras_serializable()`
- **Configuration**: Complete `get_config()` implementations for all classes
- **Explicit Building**: Weights created in `build()` methods following Keras best practices
- **Model Saving**: Full support for `model.save()` and `keras.models.load_model()`

### API Design
- **State Management**: Explicit `carry` dictionary pattern for state persistence
- **Single-Step Call**: `call(carry, batch)` performs one reasoning step
- **External Loop**: Training script manages the ACT loop externally

### Parameter Management
- **Original**: Configuration dictionary approach
- **This Implementation**: Explicit typed parameters with defaults

## Usage

### Basic Model Creation

```python
import keras
from dl_techniques.models.trm.model import TinyRecursiveReasoningModel

# Create model with explicit parameters
model = TinyRecursiveReasoningModel(
    vocab_size=12,
    hidden_size=512,
    num_heads=8,
    expansion=4.0,
    seq_len=900,  # e.g., 30x30 grid
    puzzle_emb_len=16,
    h_layers=2,
    l_layers=2,
    halt_max_steps=16,
    halt_exploration_prob=0.1,
    no_act_continue=True,
)
```

### Training Loop Example

```python
import tensorflow as tf
import keras

# Create sample batch
batch = {
    "inputs": tf.random.uniform(shape=(32, 900), maxval=12, dtype=tf.int32),
}

# Initialize state
carry = model.initial_carry(batch)

# Run ACT loop
max_steps = 16
all_outputs = []

for step in range(max_steps):
    # Perform one reasoning step
    carry, outputs = model(carry, batch, training=True)
    all_outputs.append(outputs)
    
    # Check if all sequences have halted
    if tf.reduce_all(carry["halted"]):
        print(f"All sequences halted at step {step + 1}")
        break

# Process accumulated outputs
# Typically: weighted average of logits based on halting probabilities
```

### Model Serialization

```python
# Save model
model.save("trm_model.keras")

# Load model
loaded_model = keras.models.load_model("trm_model.keras")

# Verify configuration is preserved
assert loaded_model.hidden_size == 512
assert loaded_model.halt_max_steps == 16
```

### Inference Example

```python
# During inference, halting happens only at max steps
carry = model.initial_carry(batch)

for step in range(model.halt_max_steps):
    carry, outputs = model(carry, batch, training=False)

# Use final logits for predictions
final_logits = outputs["logits"]
predictions = tf.argmax(final_logits, axis=-1)
```

## Architecture Variants

The model supports different transformer architectures through configuration:

### LLaMA-Style (Pre-Norm + RMS Norm + SwiGLU)
```python
model = TinyRecursiveReasoningModel(
    vocab_size=12,
    hidden_size=512,
    num_heads=8,
    expansion=4.0,
    seq_len=900,
    normalization_position='pre',  # Pre-norm like LLaMA
    normalization_type='rms_norm',
    ffn_type='swiglu',
)
```

### Classic Transformer (Post-Norm + Layer Norm + MLP)
```python
model = TinyRecursiveReasoningModel(
    vocab_size=12,
    hidden_size=512,
    num_heads=8,
    expansion=4.0,
    seq_len=900,
    normalization_position='post',  # Post-norm like original Transformer
    normalization_type='layer_norm',
    ffn_type='mlp',
)
```

### Custom Architecture
```python
model = TinyRecursiveReasoningModel(
    vocab_size=12,
    hidden_size=512,
    num_heads=8,
    expansion=4.0,
    seq_len=900,
    attention_type='grouped_query',  # Use grouped query attention
    ffn_type='geglu',                # Use GeGLU activation
    normalization_type='layer_norm',
    normalization_position='pre',
)
```

## State and Data Flow

### Complete Flow Diagram
```
External Loop (Training Script):
  │
  ├─ Initialize: carry = model.initial_carry(batch)
  │                 ↓
  │         carry["inner_carry"] = {z_H: zeros, z_L: zeros}
  │         carry["halted"] = True (all sequences)
  │         carry["steps"] = 0
  │
  └─ For each ACT step (until all halted or max_steps):
      │
      ├─ Step 1: Call model(carry, batch, training=True)
      │    │
      │    ├─ Reset z_H, z_L for halted sequences (using H_init, L_init)
      │    │
      │    ├─ TRMInner.call():
      │    │    ├─ Embed tokens → input_emb
      │    │    ├─ Pad with puzzle embedding → input_emb_padded
      │    │    ├─ L_level: z_L = TRMReasoningModule(z_L, input_emb_padded)
      │    │    ├─ H_level: z_H = TRMReasoningModule(z_H, z_L)
      │    │    ├─ Generate: logits = lm_head(z_H)
      │    │    └─ Generate: (q_halt, q_continue) = q_head(z_H[:, 0])
      │    │
      │    ├─ Update halting mask based on:
      │    │    ├─ Max steps reached
      │    │    ├─ Learned signals (q_halt vs q_continue)
      │    │    └─ Exploration probability
      │    │
      │    └─ Return: (new_carry, outputs)
      │
      └─ Accumulate outputs and update carry
```

### Carry Structure
```python
carry = {
    "inner_carry": {
        "z_H": Tensor[batch, seq_len+puzzle_emb_len, hidden_size],
        "z_L": Tensor[batch, seq_len+puzzle_emb_len, hidden_size],
    },
    "steps": Tensor[batch],  # int32
    "halted": Tensor[batch], # bool
    "current_data": {
        "inputs": Tensor[batch, seq_len],  # Current input for each sequence
    }
}
```

### Outputs Structure
```python
outputs = {
    "logits": Tensor[batch, seq_len, vocab_size],        # Prediction logits
    "q_halt_logits": Tensor[batch],                      # Halt probability logits
    "q_continue_logits": Tensor[batch],                  # Continue probability logits
    "target_q_continue": Tensor[batch] (optional),       # Only with Q-learning
}
```

## Advanced Features

### Exploration During Training
The model includes exploration mechanisms to encourage diverse halting behavior:
- Random exploration with probability `halt_exploration_prob`
- Forces continuation to random steps between 2 and `halt_max_steps`
- Prevents premature convergence to single halting strategy

### Q-Learning Halting (Optional)
When `no_act_continue=False`:
- Uses lookahead to compute target Q-values
- Bellman-style update: `target_q = max(next_q_halt, next_q_continue)`
- Enables more sophisticated halting decisions
- Returns `target_q_continue` in outputs for loss computation

### Gradient Control
- States are detached between steps: `tf.stop_gradient(z_H)`, `tf.stop_gradient(z_L)`
- Prevents gradient flow through multiple unrolled steps
- Stabilizes training by treating each step independently

### Learnable Initial States
- `H_init` and `L_init` are trainable weights
- Allow model to learn optimal starting points for reasoning
- Reset for each new sequence when `halted=True`

## Best Practices

### 1. Start with Simple Configuration
```python
model = TinyRecursiveReasoningModel(
    vocab_size=vocab_size,
    hidden_size=256,       # Smaller for faster experimentation
    num_heads=4,
    expansion=4.0,
    seq_len=seq_len,
    h_layers=2,
    l_layers=2,
    halt_max_steps=8,      # Fewer steps for debugging
)
```

### 2. Use Simple Halting First
Start with `no_act_continue=True` before trying Q-learning halting for easier debugging.

### 3. Monitor Halting Behavior
Track average steps per sequence to ensure the model is learning appropriate computation allocation.

### 4. Accumulate Losses Properly
Weight each step's loss by halting probability for proper ACT training.

### 5. Save Regularly
The model fully supports Keras serialization:
```python
model.save("checkpoints/trm_epoch_{epoch}.keras")
```

## Citation

If you use this implementation, please cite both the original paper and acknowledge this Keras implementation:

```bibtex
@article{jolicoeur2025less,
  title={Less is More: Recursive Reasoning with Tiny Networks},
  author={Jolicoeur-Martineau, Alexia},
  journal={arXiv preprint arXiv:2510.04871},
  year={2025}
}
```

## License

This implementation follows the license of the original TRM repository. Please refer to the original repository for licensing information.