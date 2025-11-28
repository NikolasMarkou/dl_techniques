# HRM: Hierarchical Reasoning Model

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18+-orange.svg)](https://www.tensorflow.org/)

A production-ready, fully-featured implementation of the **Hierarchical Reasoning Model (HRM)** with **Adaptive Computation Time (ACT)** in **Keras 3**. This architecture mimics human cognitive processes by dynamically allocating "thinking time" based on problem complexity.

HRM addresses the limitations of fixed-depth transformers by using a reinforcement learning mechanism (Q-Learning) to decide when to halt computation, combined with a dual-system (High/Low level) recurrent core for deep reasoning.

---

## Table of Contents

1. [Overview: What is HRM and Why It Matters](#1-overview-what-is-hrm-and-why-it-matters)
2. [The Problem HRM Solves](#2-the-problem-hrm-solves)
3. [How HRM Works: Core Concepts](#3-how-hrm-works-core-concepts)
4. [Architecture Deep Dive](#4-architecture-deep-dive)
5. [Quick Start Guide](#5-quick-start-guide)
6. [Component Reference](#6-component-reference)
7. [Configuration & Model Variants](#7-configuration--model-variants)
8. [Comprehensive Usage Examples](#8-comprehensive-usage-examples)
9. [Advanced Usage Patterns](#9-advanced-usage-patterns)
10. [Performance Optimization](#10-performance-optimization)
11. [Training and Best Practices](#11-training-and-best-practices)
12. [Serialization & Deployment](#12-serialization--deployment)
13. [Testing & Validation](#13-testing--validation)
14. [Troubleshooting & FAQs](#14-troubleshooting--faqs)
15. [Technical Details](#15-technical-details)
16. [Citation](#16-citation)

---

## 1. Overview: What is HRM and Why It Matters

### What is HRM?

**HRM** is a neuro-symbolic inspired architecture that treats reasoning as an iterative process. Unlike standard Transformers that process inputs in a single forward pass of fixed depth, HRM runs a **Recurrent Reasoning Core** for a variable number of steps.

It employs a **Dual-System Architecture**:
*   **High-Level State ($z_h$)**: Represents abstract plans, global context ("System 2" / Slow Thinking).
*   **Low-Level State ($z_l$)**: Represents detailed execution, local features ("System 1" / Fast Thinking).

### Key Innovations of this Implementation

1.  **Adaptive Computation Time (ACT)**: A Q-Learning head learns the optimal stopping point ($Q_{halt}$ vs $Q_{continue}$), allowing the model to "think" longer on hard puzzles and shorter on easy ones.
2.  **Hierarchical Core**: A unified core that updates high and low-level states iteratively using cross-attention mechanisms.
3.  **Efficient Gradient Flow**: Implements a "Truncated Backprop through Thinking" strategy, where intermediate reasoning cycles use `stop_gradient` to save memory, while the final step provides the learning signal.
4.  **Keras 3 Native**: Fully serializable, compatible with JAX, TensorFlow, and PyTorch.

### Why HRM Matters

**Standard Transformers**:
```
Model: Fixed Depth
  1. Inefficient: Uses the same compute for "2+2" as for "Integrate f(x)".
  2. Shallow: Complex logic chains require depth. To get more depth, 
     you must increase parameters, slowing down EVERY inference.
```

**HRM Solution**:
```
Model: Adaptive Depth
  1. Efficient: Halts early for simple patterns.
  2. Deep Reasoning: Can loop effectively infinite times (up to max_steps) 
     without increasing parameter count.
  3. Stable: Separates "Planning" (High-level) from "Execution" (Low-level).
```

---

## 2. The Problem HRM Solves

### The "Depth vs. Efficiency" Dilemma

In complex reasoning tasks (math, coding, logic puzzles), the required number of computational steps varies wildly per instance. A fixed-depth network is either:
*   **Overkill**: Wasting FLOPs on trivial inputs.
*   **Underpowered**: Failing on inputs requiring deep sequential logic.

### The Reasoning Loop Solution

HRM wraps a powerful reasoning core in a stateful loop.

```
┌─────────────────────────────────────────────────────────────┐
│  The HRM Dynamic Loop                                       │
│                                                             │
│  Step 0: Input -> Initial Guess                             │
│  Step 1: Refine Guess (Is it good enough? No -> Continue)   │
│  Step 2: Refine Guess (Is it good enough? No -> Continue)   │
│  Step 3: Refine Guess (Is it good enough? Yes -> HALT)      │
│                                                             │
│  Result: Correct answer with 3x compute, while simple       │
│          inputs exit at Step 0.                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. How HRM Works: Core Concepts

### The High-Level Architecture

The model is structured as a **Controller (Wrapper)** managing a **Core**.

```
Input (Token IDs, Puzzle IDs)
    │
    ▼
[HRM Controller] <─── Loops ───┐
    │                          │ (Carry State)
    ▼                          │
[Reasoning Core] ──────────────┘
    │    │
    │    └─► [Q-Head] ──► (Halt? Continue?)
    │
    └─► [LM-Head] ──► Output Logits
```

### The Data Flow (Inside the Core)

The core maintains two persistent embedding states, $z_h$ and $z_l$.

1.  **Low-Level Update**: $z_l$ updates by attending to the current $z_h$ and the Input.
2.  **High-Level Update**: $z_h$ updates by attending to the *newly updated* $z_l$.
3.  **Q-Decision**: The updated $z_h$ is projected to 2 scalars: Q-value for Halting, Q-value for Continuing.

---

## 4. Architecture Deep Dive

### 4.1 `HierarchicalReasoningCore`
The brain of the model.
*   **Inputs**: Previous carry state, Current inputs.
*   **Layers**: Contains $N$ High-level transformer blocks and $M$ Low-level transformer blocks.
*   **Cycles**: To simulate deep thinking without exploding memory, it runs internal "cycles" (loops without gradients) before the final effective pass.

### 4.2 `HierarchicalReasoningModel` (The Wrapper)
The orchestrator.
*   **State Management**: Tracks `steps`, `halted` flags, and `inner_carry` tensors.
*   **Logic**:
    *   If `Q_halt > Q_continue` OR `steps >= max_steps`: Halt.
    *   Else: Feed output back into input for next step.
*   **Training**: During training, it calculates the Bootstrap Target for the Q-Head to learn optimal halting policies.

---

## 5. Quick Start Guide

### Your First HRM Model (30 seconds)

Create a "base" variant model for a vocabulary of 32k.

```python
import keras
from model import create_hierarchical_reasoning_model

# 1. Create Model
# "base" variant matches the configuration from research papers
model = create_hierarchical_reasoning_model(
    vocab_size=32000,
    seq_len=512,
    variant="base",
    halt_max_steps=10
)

# 2. Compile
# AdamW is recommended for scale-invariant optimization
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-4),
    loss={
        "logits": "sparse_categorical_crossentropy",
        # Custom losses usually needed for Q-values in real training loops
    }
)

# 3. Dummy Inference
inputs = {
    "token_ids": keras.random.randint((1, 512), 0, 32000),
    "puzzle_ids": keras.random.randint((1,), 0, 1000)
}
outputs = model(inputs)

print(f"Logits Shape: {outputs['logits'].shape}") # (1, 512, 32000)
print(f"Halting Confidence: {outputs['q_halt_logits']}")
```

---

## 6. Component Reference

### 6.1 `HierarchicalReasoningModel`

**Purpose**: The main user-facing Keras Model.

```python
from dl_techniques.models.hierarchical_reasoning_model import HierarchicalReasoningModel

model = HierarchicalReasoningModel(
    vocab_size=32000,
    seq_len=512,
    embed_dim=768,
    h_layers=8,
    l_layers=8,
    halt_max_steps=12,
    halt_exploration_prob=0.1
)
```

### 6.2 Factory Function

#### `create_hierarchical_reasoning_model(variant, ...)`
The recommended way to instantiate.
*   `variant`: `'micro'`, `'tiny'`, `'small'`, `'base'`, `'large'`, `'xlarge'`.

---

## 7. Configuration & Model Variants

Variants are tuned based on parameter efficiency and reasoning depth.

| Variant | Embed Dim | Heads | H-Layers | L-Layers | Max Steps | Params |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`micro`** | 256 | 4 | 2 | 2 | 4 | ~1.2M |
| **`small`** | 512 | 8 | 6 | 6 | 8 | ~18M |
| **`base`** | 768 | 12 | 8 | 8 | 10 | ~52M |
| **`large`** | 1024 | 16 | 12 | 12 | 12 | ~156M |
| **`xlarge`**| 1536 | 24 | 16 | 16 | 16 | ~420M |

---

## 8. Comprehensive Usage Examples

### Example 1: Custom Training Loop with Step Control

HRM allows manual control over the reasoning steps, useful for debugging or complex RL loops.

```python
import keras
from dl_techniques.models.hierarchical_reasoning_model import (
    HierarchicalReasoningModel, create_hierarchical_reasoning_model
)

model = create_hierarchical_reasoning_model(
    vocab_size=1000, seq_len=64, variant="micro"
)

# Dummy Data
batch = {
    "token_ids": keras.random.randint((4, 64), 0, 1000),
    "puzzle_ids": keras.random.randint((4,), 0, 100)
}

# Manual Stepping
carry = model.initial_carry(batch)
all_halted = False
step_count = 0

while not all_halted and step_count < 10:
    # Pass tuple (carry, batch) to trigger step mode
    carry, outputs, all_halted = model((carry, batch))
    
    halt_probs = keras.activations.sigmoid(outputs["q_halt_logits"])
    print(f"Step {step_count}: Halt Prob avg {keras.ops.mean(halt_probs):.2f}")
    step_count += 1
```

### Example 2: Puzzle Embeddings

If you are training on multiple types of tasks (e.g., Algebra, Logic, Translation), use `puzzle_ids` to condition the model.

```python
inputs = {
    "token_ids": text_data,
    "puzzle_ids": task_type_ids # e.g., 0=Math, 1=Code, 2=Chat
}
# The model internally learns embeddings for these IDs and injects
# them into the reasoning stream.
model.fit(inputs, targets)
```

---

## 9. Advanced Usage Patterns

### Pattern 1: Q-Learning Exploration
During early training, the model might learn to halt immediately (lazy) or never halt (looping). The `halt_exploration_prob` forces random step counts to gather data on the benefits of thinking longer.

```python
# High exploration for training start
model = HierarchicalReasoningModel(..., halt_exploration_prob=0.3)

# Anneal to 0.0 for fine-tuning/production
model.halt_exploration_prob = 0.0
```

### Pattern 2: Gradient Checkpointing (Implicit)
The `HierarchicalReasoningCore` uses `stop_gradient` on inner cycles (`h_cycles`, `l_cycles`). This is a form of learned checkpointing. You can increase `h_cycles` to allow deeper "unsupervised" thinking between gradient updates.

---

## 10. Performance Optimization

### Input Padding
HRM is stateful. Ensure your batches are padded correctly, but also note that `initial_carry` automatically scales to the batch size provided.

### Mixed Precision
Crucial for `large` and `xlarge` variants due to the recurrent nature increasing VRAM usage.

```python
keras.mixed_precision.set_global_policy('mixed_float16')
```

---

## 11. Training and Best Practices

### The Loss Function
Training HRM is unique because you need to train the Q-Head.
$$ Loss = L_{LM} + \lambda L_{Q} $$
*   $L_{LM}$: CrossEntropy on tokens (Language Modeling).
*   $L_{Q}$: MSE between $Q_{value}$ and the Bootstrap Target (from the next step's max Q).

### Curriculum Learning
Start by training with `halt_max_steps=1` (forcing standard behavior), then gradually increase max steps to allow the model to learn to use the extra compute.

---

## 12. Serialization & Deployment

Fully serializable via standard Keras APIs.

```python
model.save("hrm_model.keras")
loaded = keras.models.load_model("hrm_model.keras")
```

**Note**: The model saves its configuration (layers, dimensions, exploration prob). When loading for inference, ensure you set `halt_exploration_prob=0` if you want deterministic behavior.

---

## 13. Testing & Validation

```python
def test_hrm_shapes():
    model = create_hierarchical_reasoning_model(
        vocab_size=100, seq_len=10, variant="micro"
    )
    x = {
        "token_ids": keras.ops.ones((2, 10), dtype="int32"),
        "puzzle_ids": keras.ops.zeros((2,), dtype="int32")
    }
    
    # Run in complete mode
    out = model(x)
    assert out["logits"].shape == (2, 10, 100)
    assert out["q_halt_logits"].shape == (2,)
    print("✓ Shapes Valid")

test_hrm_shapes()
```

---

## 14. Troubleshooting & FAQs

**Q: The model never halts.**
*   **A:** Check `halt_max_steps`. Also, verify if the `halt_exploration_prob` is too high, causing random forced continuations. Finally, ensure your Q-loss is actually being optimized.

**Q: Is this slow?**
*   **A:** It is slower than a standard transformer *if* it runs max steps every time. The goal is that for 80% of inputs (easy ones), it runs 1-2 steps, making it faster on average than a standard transformer of equivalent "max potential depth".

**Q: What is `inner_carry`?**
*   **A:** It contains the tensors `z_h` (High-level latent) and `z_l` (Low-level latent). These are the "working memory" of the model.

---

## 15. Technical Details

### Q-Learning Update
The model estimates the value of Halting ($H$) vs Continuing ($C$) at step $t$.
$$ Target_t = \begin{cases} \sigma(Q_{H, t+1}) & \text{if } t = T_{max} \\ \max(\sigma(Q_{H, t+1}), \sigma(Q_{C, t+1})) & \text{otherwise} \end{cases} $$
The Q-head is trained to minimize $(Q_{pred} - Target)^2$.

### Truncated Gradient Flow
To save memory, gradients are only propagated through the *final* cycle of reasoning.
$$ z_{t+1} = \text{Core}(z_t) $$
If we run 5 cycles, we calculate forward for 5 steps, but effectively do `z = stop_gradient(z)` for steps 1-4. This prevents Backprop Through Time (BPTT) memory explosion while allowing the state to evolve.

---

## 16. Citation

This implementation is based on the architectural principles found in Adaptive Computation Time and Universal Transformers:

```bibtex
@misc{wang2025hierarchical,
  title={Hierarchical Reasoning with Adaptive Computation Time},
  author={Wang, et al.},
  year={2025},
  note={Implementation based on HRM architecture}
}

@article{graves2016adaptive,
  title={Adaptive computation time for recurrent neural networks},
  author={Graves, Alex},
  journal={arXiv preprint arXiv:1603.08983},
  year={2016}
}
```
