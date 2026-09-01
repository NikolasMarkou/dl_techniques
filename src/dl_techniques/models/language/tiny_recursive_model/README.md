# Tiny Recursive Model (TRM)

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18%2B-orange.svg)](https://www.tensorflow.org/)

A Keras 3 implementation of the **Tiny Recursive Model (TRM)**, a parameter-efficient architecture for reasoning tasks. TRM uses a small shared network applied recursively, refining its solution over a variable number of steps.

This adapts the PyTorch model from "[Less is More: Recursive Reasoning with Tiny Networks](https://arxiv.org/abs/2510.04871)" to Keras 3, including **Adaptive Computation Time (ACT)**, which lets the model learn how many "thinking" steps a given problem needs.

---

## 1. Overview: What is TRM and Why It Matters

**TRM** performs reasoning by repeatedly applying a small, shared network. Instead of stacking many unique layers, it uses a shallow network and iterates, feeding its own state back as input. Depth becomes a property of the *computation*, not of the *architecture*.

### Key Innovations

1.  **Recursive reasoning.** A compact set of Transformer layers is applied many times, building a deep computational graph without a matching growth in parameter count.
2.  **Adaptive Computation Time.** TRM does not run for a fixed number of steps. A halting head decides at each step whether the current solution is good enough, so easy problems finish quickly and hard ones get more compute.
3.  **Hierarchical latent states.** The model carries two connected states: a high-level `z_H` for abstract reasoning and a low-level `z_L` for detail.

The architecture suits algorithmic puzzles (Sudoku, mathematical reasoning), multi-step logical deduction, and reasoning under tight memory budgets.

---

## 2. The Problem TRM Solves

Scaling laws push toward ever-larger feed-forward models. That has two costs. The obvious one is hardware. The subtler one is that computation is **static**: a trivial input costs exactly as much to process as a hard one, because the same stack of layers runs either way.

Recurrent networks solve the parameter half of that problem by reusing weights over time, but they historically struggled with long-range dependencies and still spent a fixed amount of computation per step.

TRM takes a third path. It keeps modern Transformer blocks for their modelling power, reuses them recursively for parameter efficiency, and adds ACT so the *number* of applications adapts to the input:

```
Instead of:  data -> Layer 1 -> Layer 2 -> ... -> Layer 100 -> output
TRM does:    data -> Model -> state1 -> Model -> state2 -> ... -> output
                              (same weights every time)
```

An ACT mechanism then learns when to stop, tailoring the computational budget to the difficulty of the problem.

---

## 3. How TRM Works: Core Concepts

### The two loops

1.  **Outer loop (ACT).** Managed by *your* training script or inference driver, not by the model. It calls the model repeatedly, threading the `carry` state from one step to the next, and stops when the halting signal fires or `halt_max_steps` is reached. At both training and inference the same learned signal is consulted (`q_halt > 0`, or `q_halt > q_continue` under Q-learning mode). Training additionally applies an exploration branch that can force continuation for a random subset of sequences; inference does not.
2.  **Inner loop (reasoning cycle).** A fixed two-stage update inside `TRMInner`, run *once* per outer step, which updates `z_L` and then `z_H`.

```
External driver (the OUTER loop)
┌────────────────────────────────────────────────────┐
│ carry = model.initial_carry(batch)                 │
│ FOR step in 1..halt_max_steps:                     │
│     carry, outputs = model(carry, batch)           │
│     IF all(carry["halted"]): BREAK                 │
└────────────────────────────────────────────────────┘
                         │
                         ▼
Inside model.call() (ONE thought step)
┌────────────────────────────────────────────────────┐
│ 1. Reset z_H / z_L for sequences marked halted     │
│ 2. TRMInner (the INNER loop):                      │
│      a. z_L <- L_level(z_L, input_emb)             │
│      b. z_H <- H_level(z_H, z_L)                   │
│ 3. logits = lm_head(z_H); q = q_head(z_H[:, 0])    │
│ 4. Update the `halted` mask, return the new carry  │
└────────────────────────────────────────────────────┘
```

### The single-step data flow

**Inputs** are the `carry` from the previous step (`z_H`, `z_L`, `steps`, `halted`, `current_data`) and the input `batch`.

1.  **State management.** Where `carry["halted"]` is `True`, reset that sequence's `z_H` and `z_L` to the learnable `H_init` and `L_init` weights, and refresh `current_data` with the new input.
2.  **Inner reasoning.** Embed the tokens, then `z_L_new = L_level(z_L_old, input_emb)` followed by `z_H_new = H_level(z_H_old, z_L_new)`.
3.  **Output generation.** `logits = lm_head(z_H_new)`, and `(q_halt, q_continue) = q_head(z_H_new[:, 0])`.
4.  **Halting and state update.** Increment `steps`, update the `halted` mask from `q_halt`, `q_continue` and `halt_max_steps`, then apply `keras.ops.stop_gradient` to `z_H_new` and `z_L_new` before storing them in the new carry, which prevents backpropagation through time.

---

## 4. Architecture Deep Dive

### 4.1 `TRMReasoningModule`

A reusable stack of `TransformerLayer` instances. It takes a latent state and an input-injection tensor, adds them, and runs the sum through its stack. Both `H_level` and `L_level` are instances of this class, differing only in layer count (`h_layers`, `l_layers`).

### 4.2 `TRMInner`

One full fixed-cycle reasoning step, owning the two-level state update:

1.  **Token embedding**: `Embedding` from input IDs to vectors.
2.  **`L_level`**: updates the low-level `z_L` state, injecting the embedded input.
3.  **`H_level`**: updates the high-level `z_H` state from the `L_level` output.
4.  **LM head**: a `Dense` projecting `z_H` to vocabulary logits.
5.  **Q head**: a `Dense` projecting the first token of `z_H` to the halt and continue logits.

This layer also owns the learnable **`H_init`** and **`L_init`** weights, the "blank slate" state a sequence resets to. They are learned, not zeros.

### 4.3 `TRM` (the model)

The `keras.Model` you interact with. It defines the `carry` dictionary structure, creates the starting state via `initial_carry`, and implements **one** reasoning step in `call`: reset handling, the `TRMInner` call, and the halting logic. It does not own the loop; you do (§8).

---

## 5. Quick Start Guide

### Installation

```bash
pip install keras>=3.0 tensorflow>=2.16 numpy
```

### Your First TRM Model

```python
import keras
import numpy as np

from dl_techniques.models.language.tiny_recursive_model.model import TRM

# 1. Create a TRM model
model = TRM(
    vocab_size=12,
    hidden_size=256,
    num_heads=4,
    expansion=4.0,
    seq_len=100,
    halt_max_steps=8
)

# 2. Compile (the loss is handled externally in the training loop)
model.compile(optimizer="adam")
model.summary()

# 3. A dummy batch
batch_size = 16
dummy_batch = {
    "inputs": np.random.randint(0, 12, size=(batch_size, 100)),
}

# 4. Initialize the ACT loop state.
#    Measured keys: ['current_data', 'halted', 'inner_carry', 'steps']
carry = model.initial_carry(dummy_batch)
print(f"Initial step count: {carry['steps'][0].numpy()}")   # 0

# 5. Run a single reasoning step
new_carry, outputs = model(carry, dummy_batch, training=True)
print(f"New step count: {new_carry['steps'][0].numpy()}")   # 1
print(f"Logits shape: {outputs['logits'].shape}")           # (16, 100, 12)
print(f"Any halted: {new_carry['halted'].numpy().any()}")
```

---

## 6. Component Reference

### 6.1 Model and layers

| Component | Location | Purpose |
| :--- | :--- | :--- |
| **`TRM`** | `...tiny_recursive_model.model` | The `keras.Model`. Manages carry state and executes one reasoning step per call. |
| **`TRMInner`** | `...tiny_recursive_model.components` | Performs the core two-level (`z_L`, `z_H`) reasoning for one step. |
| **`TRMReasoningModule`** | `...tiny_recursive_model.components` | A stack of `TransformerLayer` instances, used for both H- and L-level processing. |
| **`TransformerLayer`** | `...layers.transformers` | The configurable block powering `TRMReasoningModule`. |

`create_trm(...)` is a thin factory taking the same arguments as the constructor plus `name`.

### 6.2 Output schema by mode

`model(carry, batch, training=...)` always returns `(new_carry, outputs)`. The `outputs` keys depend on mode and configuration (all four rows measured):

| Mode | `no_act_continue` | Keys in `outputs` |
|------|-------------------|---------------------------|
| training | `True` (default) | `logits`, `q_halt_logits`, `q_continue_logits` |
| training | `False` (Q-learning) | the above plus `target_q_continue` (sigmoided, stop-gradient'd Bellman target) |
| inference | either | `logits`, `q_halt_logits`, `q_continue_logits` |

`HRMLoss` (in `dl_techniques.losses.hrm_loss`) consumes this schema unchanged and tolerates the absence of `target_q_continue` under `no_act_continue=True`.

---

## 7. Configuration & Model Variants

TRM ships **no named variants**: there is no `MODEL_VARIANTS` table, because the architecture is defined entirely by its constructor arguments. The parameters that matter most:

| Parameter | Controls |
| :--- | :--- |
| `hidden_size`, `num_heads`, `expansion` | Size and capacity of the core Transformer blocks. |
| `h_layers`, `l_layers` | Depth of each reasoning module, so how powerful one "thought" step is. Both default to 2. |
| `halt_max_steps` | Maximum computational depth (default 10). |
| `puzzle_emb_len` | Length of the prefix reserved for puzzle embeddings (default 16). |
| `no_act_continue` | `True` (default) for simple halting, `False` for the Q-learning mechanism (§9). |

Component choices pass through to `TransformerLayer`, so you can pick an architectural style directly. The defaults are already modern: `attention_type='group_query'`, `ffn_type='swiglu'`, `normalization_type='rms_norm'`.

```python
from dl_techniques.models.language.tiny_recursive_model.model import TRM

# Closer to the original "Attention Is All You Need" block
classic_trm = TRM(
    vocab_size=32000,
    hidden_size=512,
    num_heads=8,
    expansion=4.0,
    seq_len=1024,
    normalization_position='post',
    normalization_type='layer_norm',
    ffn_type='mlp',
)
```

---

## 8. Comprehensive Usage Examples

### A complete external training loop

Because the number of steps depends on the data, the loop lives outside the model. This unrolls the reasoning process, weights each step's loss by the probability of halting there, and applies the gradients.

```python
import keras
import numpy as np
import tensorflow as tf
from dl_techniques.models.language.tiny_recursive_model.model import TRM

model = TRM(vocab_size=12, hidden_size=64, num_heads=2,
            expansion=2.0, seq_len=50, halt_max_steps=4)
dummy_batch = {"inputs": np.random.randint(0, 12, size=(8, 50))}
dummy_labels = dummy_batch["inputs"]   # for simplicity, reconstruct the input

optimizer = keras.optimizers.Adam()
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

with tf.GradientTape() as tape:
    carry = model.initial_carry(dummy_batch)
    all_step_outputs = []

    # Unroll the reasoning process
    for step in range(model.halt_max_steps):
        carry, outputs = model(carry, dummy_batch, training=True)
        all_step_outputs.append(outputs)
        if tf.reduce_all(carry["halted"]):   # every sequence is done
            break

    # ACT loss. Simplified: a full objective also carries a ponder cost (section 11).
    total_loss = 0.0
    p_continue = 1.0
    for outputs in all_step_outputs:
        step_loss = loss_fn(dummy_labels, outputs["logits"])
        q_probs = keras.ops.softmax(
            keras.ops.stack(
                [outputs["q_halt_logits"], outputs["q_continue_logits"]], axis=-1
            )
        )
        p_halt_step = q_probs[:, 0]
        total_loss += tf.reduce_mean(p_continue * p_halt_step * step_loss)
        p_continue = p_continue * (1.0 - p_halt_step)

grads = tape.gradient(total_loss, model.trainable_variables)
optimizer.apply_gradients(zip(grads, model.trainable_variables))

print(f"Loop ran for {len(all_step_outputs)} steps, loss {total_loss.numpy():.4f}")
```

---

## 9. Advanced Usage Patterns

### Pattern 1: Q-learning for halting decisions

Setting `no_act_continue=False` trains the halting head as a Q-function instead of a plain gate. In training mode the `outputs` dict then carries an extra `target_q_continue` key (§6.2), and you add a Bellman-style term to the loss:

```python
from dl_techniques.models.language.tiny_recursive_model.model import TRM

q_learning_model = TRM(
    vocab_size=12, hidden_size=64, num_heads=2,
    expansion=2.0, seq_len=50,
    no_act_continue=False,
)

# Inside your training loop:
#   carry, outputs = q_learning_model(carry, batch, training=True)
#   q_loss = keras.losses.binary_crossentropy(
#       outputs["target_q_continue"], outputs["q_continue_logits"], from_logits=True
#   )
#   total_loss += tf.reduce_mean(q_loss)
```

This trains the model to predict the expected value of continuing, which gives a more globally sensible halting policy than a myopic gate.

---

## 10. Performance Optimization

TRM is built from Transformer layers, so mixed precision applies in the usual way.

```python
keras.mixed_precision.set_global_policy('mixed_float16')

model = TRM(...)   # picks up the global policy
```

In a custom training loop (which TRM requires) `model.fit()` is not doing loss scaling for you, so wrap the optimizer yourself: `optimizer = keras.mixed_precision.LossScaleOptimizer(optimizer)`.

---

## 11. Training and Best Practices

### The ACT loss and ponder cost

A proper ACT objective has two parts:

1.  **Prediction loss**: the task loss, weighted at each step by the halting probabilities, as in §8.
2.  **Ponder cost**: a penalty on the number of steps taken, so `Loss = PredictionLoss + ponder_penalty * N_steps`. Without it the model has no reason to stop early.

### Monitoring

Log the average number of steps per batch. It should start high and fall as the model learns. If it pins to `halt_max_steps` forever, the ponder penalty is too low, or the task is too hard for the current capacity.

### Start simple

Begin with `no_act_continue=True` and a small `halt_max_steps` (4 or 8) to confirm your loop and loss are right before adding the Q-learning machinery.

---

## 12. Serialization & Deployment

`TRM`, `TRMInner` and `TRMReasoningModule` are fully serializable in the `.keras` format. Each is registered with `@register_dl_technique(...)` from `dl_techniques.utils.keras_registration`: `TRM` under `dl_techniques.models.tiny_recursive_model.model`, the two layers under `...tiny_recursive_model.components`, which is the defining module's dotted path with the `language/` family directory stripped.

```python
model = TRM(...)
# ... training loop ...

model.save('my_trm_model.keras')

# No custom_objects needed.
loaded_model = keras.models.load_model('my_trm_model.keras')
assert loaded_model.hidden_size == model.hidden_size
```

---

## 13. Testing & Validation

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_models/test_tiny_recursive_model/ -q
```

A minimal state-transition check of your own:

```python
import tensorflow as tf
from dl_techniques.models.language.tiny_recursive_model.model import TRM

def test_single_step_execution():
    model = TRM(vocab_size=12, hidden_size=64, num_heads=2, expansion=2.0, seq_len=50)
    batch = {"inputs": tf.zeros((4, 50), dtype=tf.int32)}

    carry = model.initial_carry(batch)
    assert tf.reduce_all(carry["halted"])       # every sequence starts halted
    assert tf.reduce_all(carry["steps"] == 0)

    new_carry, outputs = model(carry, batch, training=False)
    assert tf.reduce_all(new_carry["steps"] == 1)
    assert outputs["logits"].shape == (4, 50, 12)
```

---

## 14. Troubleshooting & FAQs

-   **The model always runs for `halt_max_steps`.** It has not learned to halt. Raise the ponder cost, check that your ACT loss really weights each step's output, or give the model more capacity (`hidden_size`, `h_layers`, `l_layers`).
-   **`model.fit()` does not work.** Expected. `fit` assumes one input batch produces one output through a static graph; TRM's step count depends on the data. The external loop is what manages the persistent `carry` and accumulates per-step losses (§8).
-   **What is `carry`?** A dict holding state between steps: the latent states, the step counters, the halting mask, and the current input data per batch item. It is the memory of the reasoning process.
-   **Why `stop_gradient` between steps?** To avoid backpropagating through the whole unrolled sequence. Each step is trained to improve the *next* state from the *current* one, which sidesteps the vanishing and exploding gradients that make deep recurrence hard to train.
-   **What are `H_init` / `L_init`?** Learnable tensors giving the model's initial "blank slate" thought state. A resetting sequence starts from these learned vectors rather than zeros.

### Known limitations

-   **Puzzle embeddings are zero-padded.** The `puzzle_emb_len` prefix is filled with zeros rather than driven by a sparse learnable puzzle-embedding table, as in the HRM and original TRM PyTorch implementations. This is a deviation from the paper.
-   **`current_data` starts zero-filled.** `initial_carry` sets `current_data` to zeros with `halted=True`, so the first call replaces them with the real batch through the reset-on-halt branch. Token id 0 may appear transiently in `current_data` before that reset. This is intended behaviour.
-   **Mixed precision is untested.** The code is mixed-precision compatible by construction (the initial carry uses `self.compute_dtype`), but the `mixed_float16` path carries no smoke test.

---

## 15. Technical Details

### Gradient control

Inside `TRMInner.call`, the updated `z_H` and `z_L` are wrapped in `keras.ops.stop_gradient` before entering the new carry. The optimizer therefore only sees the computation within the current step: the model learns to produce a good output *and* a good next state, without a gradient path back through every previous state.

### Q-learning halting

With `no_act_continue=False`, the halting head is trained as a Q-function, where `q_halt` is `Q(s, halt)` and `q_continue` is `Q(s, continue)`. A one-step lookahead gives the value of the next state, `V(s') = max(q_halt', q_continue')`, and the target for `q_continue` is `r + γ · V(s')`. Here the reward `r` is implicitly 0 and the discount `γ` is 1, so the target reduces to `V(s')`.

---

## 16. Citation

```bibtex
@article{jolicoeur2025less,
  title={Less is More: Recursive Reasoning with Tiny Networks},
  author={Jolicoeur-Martineau, Alexia},
  journal={arXiv preprint arXiv:2510.04871},
  year={2025}
}
```

Original PyTorch repository: [samsungsailmontreal/tinyrecursivemodels](https://github.com/samsungsailmontreal/tinyrecursivemodels)

Foundational work:

```bibtex
@article{wang2025hierarchical,
  title={Hierarchical Reasoning Model},
  author={Wang, G, et al.},
  journal={arXiv preprint arXiv:2506.21734},
  year={2025}
}

@inproceedings{graves2016adaptive,
  title={Adaptive computation time for recurrent neural networks},
  author={Graves, Alex},
  booktitle={Advances in neural information processing systems},
  year={2016}
}
```
