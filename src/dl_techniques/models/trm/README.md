# Tiny Recursive Model (TRM)

This repository contains a Keras 3 implementation of the **Tiny Recursive Model (TRM)**, a highly parameter-efficient architecture for complex reasoning tasks. This implementation is a direct adaptation of the original PyTorch model from the paper "[Less is More: Recursive Reasoning with Tiny Networks](https://arxiv.org/abs/2510.04871)".

The core idea of TRM is to achieve powerful reasoning capabilities not by scaling up model size, but by recursively applying a small, shared neural network to refine its internal state and predictions over a variable number of steps. This approach is inspired by the principles of **Adaptive Computation Time (ACT)**, allowing the model to dynamically allocate more "thinking" time to more difficult problems.

This implementation prioritizes clarity, modularity, and adherence to Keras best practices, making it easy to integrate, understand, and serialize within the TensorFlow ecosystem.

### References and Original Work

This work would not be possible without the foundational research and open-source contributions of the original authors. Please cite their work if you use this model.

-   **Paper**: Jolicoeur-Martineau, A. (2025). *Less is More: Recursive Reasoning with Tiny Networks*. [arXiv:2510.04871](https://arxiv.org/abs/2510.04871).
-   **Original PyTorch Repository**: [samsungsailmontreal/tinyrecursivemodels](https://github.com/sapientinc/HRM) (Note: The original code is based on the HRM repository).
-   **Inspiration**: Wang, G., et al. (2025). *Hierarchical Reasoning Model*. [arXiv:2506.21734](https://arxiv.org/abs/2506.21734).

## Core Concepts

The model is built on three fundamental principles:

1.  **Recursive Reasoning**: Instead of a deep, feed-forward network, TRM uses a small, shallow network (the `TRMReasoningModule`) that is called repeatedly. At each step, it takes its own previous output (its latent state) as input, allowing it to progressively refine its "thought process." This makes the model extremely parameter-efficient.

2.  **Adaptive Computation Time (ACT)**: The model learns *how many* recursive steps are needed for a given problem. At each step, a dedicated "halting head" (`q_head`) outputs a probability of whether to stop or continue reasoning. This allows the model to save computation on easy examples and dedicate more resources to harder ones.

3.  **Hierarchical Latent States**: The model maintains two levels of latent states, `z_H` (high-level) and `z_L` (low-level). Within a single "thought" step (one call to the model's `forward` method), these states are updated through a series of nested cycles (`H_cycles` and `L_cycles`), representing a focused burst of computation before the model decides whether to halt or continue to the next step.

## Architectural Breakdown

The implementation in `model.py` is structured into three main Keras classes, promoting modularity and reusability.

### 1. `TRMReasoningModule` (`keras.layers.Layer`)

This is the most basic building block.
-   **Purpose**: A simple container that stacks `L_layers` instances of a `TransformerLayer`.
-   **Function**: It forms the core computational engine used for all state updates. It takes a latent state and an "input injection" and processes them through a sequence of transformer blocks.

### 2. `TRMInner` (`keras.layers.Layer`)

This layer encapsulates one full, multi-cycle "thought" step.
-   **Purpose**: To orchestrate the complex update logic for the `z_H` and `z_L` states.
-   **Function**:
    1.  It takes the previous step's `z_H` and `z_L` states and the current input data embeddings.
    2.  It executes a nested loop: `H_cycles` iterations of an outer loop, each containing `L_cycles` iterations of an inner loop.
    3.  Both loops reuse the same `TRMReasoningModule` (`self.L_level`) to update the states. Most of this process runs without gradient flow (`tf.stop_gradient`) to stabilize training, with only the final iteration contributing to the gradients.
    4.  After the cycles, it produces the final outputs for the current step: token `logits` for the prediction and `q_logits` for the halting decision.

### 3. `TinyRecursiveReasoningModel` (`keras.Model`)

This is the main, user-facing model.
-   **Purpose**: To manage the outer ACT loop and the persistent state (`carry`) across multiple steps.
-   **Function**:
    1.  Its `call` method represents **a single step** in the adaptive computation process.
    2.  It handles state management:
        -   **Resetting**: Resets the `z_H` and `z_L` states for batch items that have just started or finished a previous problem.
        -   **Updating**: Passes the current states to its `TRMInner` submodule for computation.
        -   **Halting**: Implements the logic to decide which batch items should halt based on the `q_logits`, the maximum step count, and exploration probabilities during training.
    3.  It is designed to be called repeatedly by an external training loop until all items in a batch have halted.

## Key Differences from the Original Implementation

This implementation stays true to the logic of the original TRM but adapts it to the Keras/TensorFlow paradigm.

-   **Framework**: The original model is written in **PyTorch**, while this version is a native **Keras 3** model, designed to run on a TensorFlow backend.
-   **Modularity**: The architecture is broken down into distinct `keras.layers.Layer` and `keras.Model` classes, which is idiomatic for Keras and improves code organization compared to the single-file, nested-class structure in the original.
-   **Keras-Native Features**:
    -   **Serialization**: All custom layers are registered with `@keras.saving.register_keras_serializable()` and implement `get_config()`, allowing the entire model to be saved and loaded using `model.save()` and `keras.models.load_model()`.
    -   **Explicit Building**: Layers follow the best practice of creating weights in the `build()` method, which is explicitly called to ensure robust model loading and state initialization.
-   **API**: The model's `call` signature (`carry`, `batch`) is designed for explicit state management, common in functional-style loops (like `tf.while_loop`) or simple Python loops in the training script.

## Usage

The `TinyRecursiveReasoningModel` is designed to be used in a training loop that manages the recursive state (`carry`). The model's `call` method performs only one step of the process.

Here is a conceptual example of how to use the model in a custom training loop:

```python
import tensorflow as tf
import keras
from model import TinyRecursiveReasoningModel

# 1. Define model configuration
config = {
    "hidden_size": 512,
    "expansion": 4.0,
    "num_heads": 8,
    "seq_len": 900,  # Example for 30x30 grid
    "puzzle_emb_len": 16,
    "L_layers": 2,
    "H_cycles": 3,
    "L_cycles": 6,
    "vocab_size": 12,
    "num_puzzle_identifiers": 1000,
    "puzzle_emb_ndim": 512,
    "halt_max_steps": 16,
    "halt_exploration_prob": 0.1,
    "no_ACT_continue": True,
}

# 2. Instantiate the model
model = TinyRecursiveReasoningModel(config)

# 3. Create a sample batch of data (shapes are illustrative)
batch = {
    "inputs": tf.random.uniform(shape=(32, 900), maxval=12, dtype=tf.int32),
    "puzzle_identifiers": tf.random.uniform(shape=(32,), maxval=1000, dtype=tf.int32),
    "labels": tf.random.uniform(shape=(32, 900), maxval=12, dtype=tf.int32),
}

# 4. Initialize the state ("carry")
carry = model.initial_carry(batch)

# 5. Run the outer ACT loop (conceptual)
# In a real scenario, you would manage losses and gradients.
total_steps = config["halt_max_steps"]
accumulated_outputs = []

for step in range(total_steps):
    # Perform one reasoning step
    new_carry, outputs_at_step = model(carry, batch, training=True)
    
    # Update the carry for the next iteration
    carry = new_carry
    
    # You would typically accumulate losses based on a halting mask
    # For simplicity, we just store outputs here.
    accumulated_outputs.append(outputs_at_step)
    
    # Check if all items in the batch have halted
    if tf.reduce_all(carry["halted"]):
        print(f"All sequences halted at step {step + 1}.")
        break

# The final prediction would be a weighted average of logits from each step,
# weighted by the halting probabilities.
print("Finished ACT loop.")

```