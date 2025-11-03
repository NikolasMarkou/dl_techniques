# Tiny Recursive Model (TRM)

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18%2B-orange.svg)](https://www.tensorflow.org/)

A production-ready, fully-featured Keras 3 implementation of the **Tiny Recursive Model (TRM)**, a highly parameter-efficient architecture designed for complex reasoning tasks. TRM challenges the "bigger is better" paradigm by using a small, shared neural network that is applied recursively to refine its solution over a variable number of steps.

This implementation adapts the original PyTorch model from the paper "[Less is More: Recursive Reasoning with Tiny Networks](https://arxiv.org/abs/2510.04871)" to Keras 3. It incorporates the principles of **Adaptive Computation Time (ACT)**, allowing the model to dynamically learn how many "thinking" steps are needed for a given problem. The code adheres to modern Keras best practices, ensuring it is modular, well-documented, and fully serializable.

---

## Table of Contents

1. [Overview: What is TRM and Why It Matters](#1-overview-what-is-trm-and-why-it-matters)
2. [The Problem TRM Solves](#2-the-problem-trm-solves)
3. [How TRM Works: Core Concepts](#3-how-trm-works-core-concepts)
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

## 1. Overview: What is TRM and Why It Matters

### What is TRM?

The **Tiny Recursive Model (TRM)** is a novel architecture that performs complex reasoning by repeatedly applying a small, shared neural network. Instead of having a massive number of unique layers (like in a standard deep Transformer), TRM uses a shallow network and iterates, feeding its own output back as input. This recursive process allows it to progressively deepen its "thought process" on a given problem.

### Key Innovations

1.  **Recursive Reasoning**: The core principle is parameter reuse. A compact set of Transformer layers is applied multiple times, allowing the model to build complex computational graphs without a corresponding explosion in parameter count.
2.  **Adaptive Computation Time (ACT)**: TRM doesn't run for a fixed number of steps. It learns to decide *when to stop thinking*. At each step, a special "halting head" determines whether the current solution is good enough or if more computation is needed. This allows it to solve easy problems quickly and dedicate more resources to harder ones.
3.  **Hierarchical Latent States**: The model maintains two separate but connected "trains of thought": a high-level state (`z_H`) for abstract reasoning and a low-level state (`z_L`) for processing details. This structured state allows for more sophisticated reasoning.

### Why TRM Matters

**The Scaling Law Problem**:
```
Problem: Solve a highly complex reasoning task.
Standard Approach (e.g., Large Language Models):
  1. Build an extremely deep network with billions or trillions of parameters.
  2. Train it on massive datasets and hardware clusters.
  3. Limitation: This is incredibly expensive, slow, and environmentally costly.
     The model uses the same massive amount of compute for every single token.
```

**TRM's Solution**:
```
TRM Approach:
  1. Build a small, parameter-efficient network.
  2. Apply this network recursively, deepening the computation, not the architecture.
  3. Let the model learn how much computation is needed per input via ACT.
  4. Benefit: Achieves powerful reasoning with a fraction of the parameters and
     dynamically allocates compute, making it highly efficient.
```

### Real-World Impact

TRM is an excellent choice for tasks requiring deep reasoning where parameter efficiency is critical:

-   🧩 **Algorithmic Puzzles**: Solving tasks like Sudoku or mathematical reasoning problems.
-   **Planning and Logic**: Performing multi-step logical deductions.
-   **Resource-Constrained Environments**: Deploying advanced reasoning on edge devices or where memory and compute are limited.
-   🧠 **Cognitive Science Modeling**: Provides a compelling model for iterative human thought processes.

---

## 2. The Problem TRM Solves

### The Compute-vs-Capability Trade-off

Modern AI is often dominated by the principle of "scaling laws," which suggest that performance directly correlates with model size and training data. This has led to a race to build ever-larger models.

```
┌───────────────────────────────────────────────────────────────┐
│  The Dilemma of Modern Architectures                          │
│                                                               │
│  Large Feed-Forward Models (e.g., GPT-style Transformers):    │
│    - Extremely powerful due to massive parameter counts.      │
│    - Suffer from static computation: a simple query ("hello") │
│      costs the same to process as a complex legal question.   │
│    - High barrier to entry due to immense hardware needs.     │
│                                                               │
│  Recurrent Neural Networks (RNNs):                            │
│    - Parameter-efficient by reusing weights over time.        │
│    - Traditionally struggled with long-range dependencies and │
│      the fixed computation per step was not ideal.            │
└───────────────────────────────────────────────────────────────┘
```

TRM offers a third way. It combines the power of modern Transformer blocks with the parameter efficiency of recurrent processing, and adds a dynamic computation mechanism (ACT) to escape the static compute limitations of both traditional RNNs and feed-forward Transformers.

### How TRM Changes the Game

TRM demonstrates that computational depth can be a more efficient path to powerful reasoning than architectural depth.

```
┌───────────────────────────────────────────────────────────────────┐
│  The TRM Recursive Reasoning Strategy                             │
│                                                                   │
│  1. The Model: A small, shallow network (`TRMInner`).             │
│                                                                   │
│  2. The Process:                                                  │
│     - Instead of data -> Layer 1 -> ... -> Layer 100 -> Output    │
│     - TRM does: data -> Model -> state1 -> Model -> state2 ...    │
│       This creates a deep computational graph with few parameters.│
│                                                                   │
│  3. The Control:                                                  │
│     - An Adaptive Computation Time (ACT) mechanism learns to      │
│       halt this process, tailoring the computational budget       │
│       to the problem's difficulty.                                │
└───────────────────────────────────────────────────────────────────┘
```

This principled approach results in a simple, scalable, and powerful architecture that can tackle complex reasoning tasks with remarkable efficiency.

---

## 3. How TRM Works: Core Concepts

### The Two-Loop Architecture

TRM's operation is best understood as two nested loops.

1.  **Outer Loop (ACT Loop)**: This is the adaptive part, managed by the training script. It calls the model repeatedly, passing the state (`carry`) from one step to the next. It checks the model's halting signal to decide whether to continue the loop.
2.  **Inner Loop (Reasoning Cycle)**: This is a fixed, two-stage process inside the `TRMInner` layer that runs *once* per outer loop step. It updates the hierarchical latent states (`z_L` and `z_H`).

```
┌──────────────────────────────────────────────────────────────────────────┐
│                             TRM Execution Flow                           │
│                                                                          │
│  External Training Script (The "Outer Loop")                             │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ carry = model.initial_carry()                                      │  │
│  │                                                                    │  │
│  │ FOR step in 1..max_steps:                                          │  │
│  │     // This is ONE call to the main model                          │  │
│  │     carry, outputs = model(carry, batch)                           │  │
│  │     // Halt if model says so or max steps reached                  │  │
│  │     IF all(carry["halted"]): BREAK                                 │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                      │                                   │
│                                      ▼                                   │
│  Inside `model.call()` (One "Thought" Step)                              │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ 1. Reset state for new sequences                                   │  │
│  │ 2. Call `TRMInner` module (The "Inner Loop")                       │  │
│  │    a. Update z_L (low-level state) using input data                │  │
│  │    b. Update z_H (high-level state) using new z_L                  │  │
│  │ 3. Generate logits (prediction) and q_logits (halt signal)         │  │
│  │ 4. Update the `halted` mask in the `carry` and return              │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

### The Complete Data Flow (Single Step)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          TRM Single-Step Data Flow                      │
└─────────────────────────────────────────────────────────────────────────┘

INPUTS:
- `carry`: The state from the previous step {`z_H`, `z_L`, `steps`, `halted`, ...}
- `batch`: The input data for this step

STEP 1: STATE MANAGEMENT
────────────────────────
- Check `carry["halted"]` mask. For any sequence where `halted` is True,
  reset its `z_H` and `z_L` states to the learnable `H_init` and `L_init` weights.
- Update `current_data` with new inputs for the reset sequences.

STEP 2: INNER REASONING (inside `TRMInner`)
──────────────────────────────────────────
- Embed input tokens from `current_data` -> `input_emb`.
- L-level Update: `z_L_new = L_level_Module(z_L_old, input_emb)`.
- H-level Update: `z_H_new = H_level_Module(z_H_old, z_L_new)`.

STEP 3: OUTPUT GENERATION
─────────────────────────
- Prediction Head: `logits = lm_head(z_H_new)`.
- Halting Head: `(q_halt, q_continue) = q_head(z_H_new[:, 0])`.

STEP 4: HALTING LOGIC & STATE UPDATE
────────────────────────────────────
- Increment `steps` counter.
- Update `carry["halted"]` mask based on `q_halt`, `q_continue`, and `max_steps`.
- Detach gradients from `z_H_new` and `z_L_new` using `tf.stop_gradient` before
  placing them in the `new_carry` to prevent backpropagation through time.

OUTPUTS:
- `new_carry`: The updated state for the next step.
- `outputs`: Dictionary containing `{logits, q_halt_logits, ...}`.
```

---

## 4. Architecture Deep Dive

### 4.1 `TRMReasoningModule`

-   **Purpose**: A reusable block of stacked Transformer layers. This is the core computational workhorse.
-   **Implementation**: A `keras.layers.Layer` that contains a list of `TransformerLayer` instances.
-   **Functionality**: It takes a latent state and an "input injection" tensor, adds them, and processes the result through its stack of Transformer layers. Both the `H_level` and `L_level` modules are instances of this class, configured with different numbers of layers.

### 4.2 `TRMInner`

-   **Purpose**: To encapsulate one full, fixed-cycle reasoning step, managing the two-level state update.
-   **Architecture**:
    1.  **Token Embedding**: An `Embedding` layer to convert input token IDs into vectors.
    2.  **`L_level` Module**: A `TRMReasoningModule` that updates the low-level `z_L` state by incorporating the embedded input tokens.
    3.  **`H_level` Module**: A `TRMReasoningModule` that updates the high-level `z_H` state using the output of the `L_level` module.
    4.  **LM Head**: A `Dense` layer that projects the final `z_H` state to vocabulary logits for prediction.
    5.  **Q Head**: A `Dense` layer that projects the first token of `z_H` to two logits representing the halt and continue probabilities.
-   **Learnable Initial States**: Crucially, this layer owns the `H_init` and `L_init` weights, which act as the "reset" state for the reasoning process.

### 4.3 `TRM` (The Model)

-   **Purpose**: The main `keras.Model` class that the user interacts with. It manages the outer ACT loop's state and logic.
-   **Responsibilities**:
    1.  **State Management**: It defines the structure of the `carry` dictionary, which holds the persistent state between steps.
    2.  **Initialization**: The `initial_carry` method creates the starting state for a new batch.
    3.  **Single-Step Execution**: The `call` method implements the logic for a single reasoning step: it handles state resets, calls the `TRMInner` module, and applies the halting logic.

---

## 5. Quick Start Guide

### Installation

```bash
# Ensure you have the required dependencies
pip install keras>=3.0 tensorflow>=2.16 numpy
```

### Your First TRM Model (30 seconds)

Let's build a small TRM and run it for a single reasoning step.

```python
import keras
import tensorflow as tf
import numpy as np

# Local imports from your project structure
from dl_techniques.models.trm.model import TRM

# 1. Create a TRM model
model = TRM(
    vocab_size=12,
    hidden_size=256,
    num_heads=4,
    expansion=4.0,
    seq_len=100,
    halt_max_steps=8
)

# 2. Compile the model (the loss is handled externally in the training loop)
model.compile(optimizer="adam")
print("✅ TRM model created and compiled successfully!")
model.summary()

# 3. Create a dummy batch of data
batch_size = 16
dummy_batch = {
    "inputs": np.random.randint(0, 12, size=(batch_size, 100)),
}

# 4. Initialize the state for the ACT loop
carry = model.initial_carry(dummy_batch)
print(f"Initial step count: {carry['steps'][0].numpy()}")

# 5. Run a single reasoning step
new_carry, outputs = model(carry, dummy_batch, training=True)
print("\n✅ Single reasoning step complete!")
print(f"New step count: {new_carry['steps'][0].numpy()}")
print(f"Logits shape: {outputs['logits'].shape}")  # (batch_size, seq_len, vocab_size)
print(f"Halted mask: {new_carry['halted'].numpy().any()}")  # Some might halt
```

---

## 6. Component Reference

### 6.1 Model & Layers

| Component | Location | Purpose |
| :--- | :--- | :--- |
| **`TRM`** | `...trm.model` | The main Keras `Model`. Manages the ACT loop state and executes single reasoning steps. |
| **`TRMInner`** | `...trm.components` | A Keras `Layer` that performs the core two-level reasoning (`z_L`, `z_H`) for one step. |
| **`TRMReasoningModule`** | `...trm.components` | A Keras `Layer` composed of a stack of `TransformerLayer` instances. Used for both H- and L-level processing. |

### 6.2 Core Building Block

| Layer | Location | Purpose |
| :--- | :--- | :--- |
| **`TransformerLayer`** | `...layers.transformers.TransformerLayer` | The highly configurable, modern Transformer block that powers the `TRMReasoningModule`. |

---

## 7. Configuration & Model Variants

TRM does not have named variants like "base" or "large." Instead, its architecture is defined by its configuration parameters. You can create different architectural styles by mixing and matching these parameters.

### Key Architectural Parameters

-   **`hidden_size`, `num_heads`, `expansion`**: Control the size and capacity of the core Transformer blocks.
-   **`h_layers`, `l_layers`**: Control the depth of the reasoning modules within each step. Increasing these makes each "thought" step more powerful.
-   **`halt_max_steps`**: The maximum computational depth the model can achieve.

### Example: LLaMA-Style Variant (Pre-Norm, RMSNorm, SwiGLU)

```python
from dl_techniques.models.trm.model import TRM

# This configuration mimics modern LLM architectures
llama_style_trm = TRM(
    vocab_size=32000,
    hidden_size=512,
    num_heads=8,
    expansion=2.66, # Common in LLaMA-style models
    seq_len=1024,
    normalization_position='pre',  # Pre-normalization
    normalization_type='rms_norm', # RMS Normalization
    ffn_type='swiglu',             # SwiGLU FFN
)
```

### Example: Classic Transformer Variant (Post-Norm, LayerNorm, ReLU/GELU)

```python
from dl_techniques.models.trm.model import TRM

# This configuration is closer to the original "Attention Is All You Need" paper
classic_trm = TRM(
    vocab_size=32000,
    hidden_size=512,
    num_heads=8,
    expansion=4.0,
    seq_len=1024,
    normalization_position='post', # Post-normalization
    normalization_type='layer_norm',# Standard LayerNorm
    ffn_type='mlp',                # Standard MLP FFN
    # The TransformerLayer's default activation is 'gelu'
)
```

---

## 8. Comprehensive Usage Examples

### Example: A Complete External Training Loop

Because TRM uses ACT, the training loop must be managed externally to handle the variable number of steps and accumulate outputs.

```python
import tensorflow as tf
import keras

# Assume `model` and `dummy_batch` are created as in the Quick Start guide
optimizer = keras.optimizers.Adam()
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
dummy_labels = dummy_batch["inputs"] # For simplicity, use inputs as labels

# --- Start of the external ACT training loop ---
with tf.GradientTape() as tape:
    carry = model.initial_carry(dummy_batch)
    all_step_outputs = []
    
    # Unroll the reasoning process
    for step in range(model.halt_max_steps):
        carry, outputs = model(carry, dummy_batch, training=True)
        all_step_outputs.append(outputs)
        
        # Optimization: Stop if all sequences in the batch have halted
        if tf.reduce_all(carry["halted"]):
            break
            
    # --- ACT Loss Calculation ---
    # This is a simplified example. A proper ACT loss also includes a ponder cost.
    # We weight the loss at each step by the probability of not have halted yet.
    total_loss = 0.0
    p_continue = 1.0 # Probability of having reached the current step
    
    for i, outputs in enumerate(all_step_outputs):
        # Calculate cross-entropy loss for this step
        step_loss = loss_fn(dummy_labels, outputs["logits"])
        
        # Calculate halting probabilities for this step
        q_probs = keras.ops.softmax(
            keras.ops.stack(
                [outputs["q_halt_logits"], outputs["q_continue_logits"]], axis=-1
            )
        )
        p_halt_step = q_probs[:, 0]
        
        # Weight loss by the probability of halting at this step
        weighted_loss = tf.reduce_mean(p_continue * p_halt_step * step_loss)
        total_loss += weighted_loss
        
        # Update the probability of continuing to the next step
        p_continue = p_continue * (1.0 - p_halt_step)

# Apply gradients
grads = tape.gradient(total_loss, model.trainable_variables)
optimizer.apply_gradients(zip(grads, model.trainable_variables))

print(f"Loop ran for {len(all_step_outputs)} steps.")
print(f"Final loss: {total_loss.numpy():.4f}")
```

---

## 9. Advanced Usage Patterns

### Pattern 1: Q-Learning for Halting Decisions

By setting `no_act_continue=False`, you enable a more sophisticated halting mechanism based on Q-learning.

```python
# Create a model with Q-learning enabled
q_learning_model = TRM(
    # ... other params ...
    no_act_continue=False
)

# During the training loop, the `outputs` dictionary will now contain
# an extra key: "target_q_continue".
# carry, outputs = q_learning_model(carry, batch, training=True)
# target_q = outputs["target_q_continue"]
# q_continue_logits = outputs["q_continue_logits"]

# You must add a Bellman-style loss to your total loss function:
# q_loss = keras.losses.binary_crossentropy(target_q, q_continue_logits, from_logits=True)
# total_loss += tf.reduce_mean(q_loss)
```
This loss trains the model to predict the expected future value of continuing, leading to more optimal halting decisions.

---

## 10. Performance Optimization

### Mixed Precision Training

TRM is built on Transformer layers, making it an ideal candidate for mixed precision training. This can provide a significant speedup on modern GPUs.

```python
# Enable mixed precision globally before creating the model
keras.mixed_precision.set_global_policy('mixed_float16')

# Create model (will automatically use mixed precision)
model = TRM(...)
model.compile(...)

# When training, use a LossScaleOptimizer to prevent numeric underflow.
# This is handled automatically by model.fit(), but in a custom loop,
# you may need to wrap your optimizer.
# optimizer = keras.mixed_precision.LossScaleOptimizer(optimizer)
```

---

## 11. Training and Best Practices

### The ACT Loss and Ponder Cost

-   A proper ACT training objective consists of two parts:
    1.  **Prediction Loss**: The standard task loss (e.g., cross-entropy), weighted at each step by the halting probabilities, as shown in the example loop.
    2.  **Ponder Cost**: A regularization term that penalizes the model for taking too many steps. It is typically the total number of steps taken (`N_steps`). The final loss is `Loss = PredictionLoss + ponder_penalty * N_steps`. This encourages the model to be efficient.

### Monitoring Halting Behavior

-   During training, it's crucial to log the average number of steps the model takes per batch. This value should ideally start high and then decrease as the model learns to solve the task more efficiently. If it always runs to `halt_max_steps`, your ponder penalty may be too low or the task too difficult for the current model size.

### Start Simple

-   Begin with `no_act_continue=True` (simple halting) and a small `halt_max_steps` (e.g., 4 or 8) to ensure the training loop and loss calculation are correct before moving to more complex configurations.

---

## 12. Serialization & Deployment

The `TRM` model and all its custom layers (`TRMInner`, `TRMReasoningModule`) are fully serializable using Keras 3's modern `.keras` format. This is possible because each custom component is decorated with `@keras.saving.register_keras_serializable()`.

### Saving and Loading

```python
# Create and train model
model = TRM(...)
# ... training loop ...

# Save the entire model to a single file
model.save('my_trm_model.keras')

# Load the model in a new session, including its architecture, weights,
# and custom layers.
loaded_model = keras.models.load_model('my_trm_model.keras')
print("✅ TRM model loaded successfully!")
assert loaded_model.hidden_size == model.hidden_size
```

---

## 13. Testing & Validation

### Unit Tests

You can validate the implementation with simple tests to ensure model creation and state transitions work as expected.

```python
import keras
import tensorflow as tf
from dl_techniques.models.trm.model import TRM


def test_model_creation():
    """Test that a model can be created."""
    model = TRM(
        vocab_size=12, hidden_size=64, num_heads=2, expansion=2.0, seq_len=50
    )
    assert model is not None
    print("✓ TRM creation successful")


def test_single_step_execution():
    """Test that a single step updates the state correctly."""
    model = TRM(
        vocab_size=12, hidden_size=64, num_heads=2, expansion=2.0, seq_len=50
    )
    batch = {"inputs": tf.zeros((4, 50), dtype=tf.int32)}

    # Initial state
    carry = model.initial_carry(batch)
    assert tf.reduce_all(carry["halted"])
    assert tf.reduce_all(carry["steps"] == 0)

    # First step
    new_carry, outputs = model(carry, batch, training=False)
    # In inference, halts only at max steps. In training, it can halt early.
    assert not tf.reduce_all(new_carry["halted"])
    assert tf.reduce_all(new_carry["steps"] == 1)
    assert outputs["logits"].shape == (4, 50, 12)
    print("✓ Single step execution and state update are correct")


# Run tests
if __name__ == '__main__':
    test_model_creation()
    test_single_step_execution()
    print("\n✅ All tests passed!")
```

---

## 14. Troubleshooting & FAQs

**Issue 1: My model always runs for `halt_max_steps`.**

-   **Cause**: The model has not learned to halt earlier. This could be because the task is too hard, the model is too small, or the "ponder cost" penalty in your loss function is too low or absent.
-   **Solution**: Try increasing the ponder cost penalty. Ensure your ACT loss is correctly weighting the outputs of each step. You can also try increasing the model's `hidden_size` or the number of `h_layers`/`l_layers`.

### Frequently Asked Questions

**Q: Why is the training loop external? Why can't I just use `model.fit()`?**

A: `model.fit()` assumes a static computational graph where one input batch produces one output. TRM's graph is dynamic; the number of forward passes depends on the data itself. The external loop is necessary to manage the persistent state (`carry`) and accumulate the outputs and losses from each variable-length unrolling.

**Q: What is the `carry` object?**

A: The `carry` is a Python dictionary that holds the model's state between recursive steps. It contains the latent states (`z_H`, `z_L`), the step counters, the halting masks, and the current input data for each item in the batch. It's the "memory" of the reasoning process.

**Q: Why are gradients stopped between steps with `tf.stop_gradient`?**

A: This is a crucial design choice to make training stable. It prevents backpropagation through the entire unrolled sequence of steps (which can be very long). Instead, the model is trained to improve its *next* step based on the *current* state, treating each reasoning step as a distinct unit. This implementation uses the native `tf.stop_gradient` for robustness with the TensorFlow backend.

**Q: What are the `H_init` and `L_init` weights for?**

A: They are learnable tensors that represent the model's optimal "blank slate" or initial thought state. Whenever a new problem is presented (i.e., when a sequence in the batch resets), the model's latent states `z_H` and `z_L` are initialized with these learned vectors instead of just zeros.

---

## 15. Technical Details

### Gradient Control and State Management

The separation of the `carry` state between steps is critical. Inside the `TRMInner.call` method, the updated `z_H` and `z_L` states are wrapped in `tf.stop_gradient` before being placed in the `new_carry`. This means that when the optimizer computes gradients, it only looks at the computation within the *current* step. The model learns to produce a good output and a good *next state*, but the gradient path does not flow back through all previous states. This avoids the vanishing/exploding gradient problems common in traditional RNNs and makes training deep recursive models feasible.

### Q-Learning Halting Mechanism

When `no_act_continue=False`, the model's halting head is trained like a Q-function in reinforcement learning.
-   `q_continue` represents `Q(s, a=continue)`.
-   `q_halt` represents `Q(s, a=halt)`.
The model makes a one-step lookahead to compute the value of the next state, `V(s') = max(q_halt', q_continue')`. The training target for `q_continue` then becomes `r + γ * V(s')`. In this model, the reward `r` is implicitly 0, and the discount `γ` is 1, so the target for `q_continue` is simply `V(s')`. This encourages the model to learn a more globally optimal halting policy.

---

## 16. Citation

This implementation is a Keras 3 adaptation of the original TRM paper. If you use this model in your research, please cite the original work:

-   **Original Paper**:
    ```bibtex
    @article{jolicoeur2025less,
      title={Less is More: Recursive Reasoning with Tiny Networks},
      author={Jolicoeur-Martineau, Alexia},
      journal={arXiv preprint arXiv:2510.04871},
      year={2025}
    }
    ```
-   **Original PyTorch Repository**: [samsungsailmontreal/tinyrecursivemodels](https://github.com/samsungsailmontreal/tinyrecursivemodels)

Please also consider citing the inspirational and foundational works:

-   **Hierarchical Reasoning Model**:
    ```bibtex
    @article{wang2025hierarchical,
      title={Hierarchical Reasoning Model},
      author={Wang, G, et al.},
      journal={arXiv preprint arXiv:2506.21734},
      year={2025}
    }
    ```
-   **Adaptive Computation Time**:
    ```bibtex
    @inproceedings{graves2016adaptive,
      title={Adaptive computation time for recurrent neural networks},
      author={Graves, Alex},
      booktitle={Advances in neural information processing systems},
      year={2016}
    }
    ```