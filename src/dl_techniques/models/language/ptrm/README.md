# Probabilistic Tiny Recursive Model (PTRM)

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18%2B-orange.svg)](https://www.tensorflow.org/)

A Keras 3 implementation of the **Probabilistic Tiny Recursive Model (PTRM)**, a test-time compute scaling framework for Tiny Recursive Models (TRM) that introduces stochastic exploration via parallel rollouts with Gaussian noise injection.

This adapts the method from "[Probabilistic Tiny Recursive Model](https://arxiv.org/abs/XXXX.XXXXX)" by Sghaier et al. (2026) to Keras 3, building on the TRM architecture from "[Less is More: Recursive Reasoning with Tiny Networks](https://arxiv.org/abs/2510.04871)".

---

## 1. Overview: What is PTRM and Why It Matters

**PTRM** is not a new model architecture — it is an **inference procedure** applied to a pretrained TRM backbone. The innovation is purely at inference time:

1.  **Stochastic exploration.** Instead of a single deterministic rollout, PTRM runs `K` parallel rollouts, injecting Gaussian noise into the latent state at each supervision step.
2.  **Q-head selection.** Each rollout produces a candidate answer scored by the model's existing Q-head (trained as a correctness predictor for ACT halting). The highest-Q answer is selected.
3.  **No retraining required.** PTRM works with any pretrained TRM checkpoint — no fine-tuning, no task-specific augmentations.

This simple mechanism allows PTRM to escape bad basins in the latent space where deterministic TRM gets stuck, yielding substantial accuracy gains across reasoning benchmarks.

### Key Results (from paper)

| Benchmark | Deterministic TRM | PTRM (K=100) | Gain |
|-----------|------------------|--------------|------|
| Sudoku-Extreme | 87.4% | **98.75%** | +11.35 pp |
| PPBench (aggregate) | 62.6% | **91.2%** | +28.6 pp |
| Maze-Hard | 83.8% | 86.7% | +2.9 pp |
| ARC-AGI-2 (pass@1) | 7.36% | 8.47% | +1.1 pp |

PTRM achieves nearly **double the accuracy of frontier LLMs** on PPBench (91.2% vs 55.1%) at **less than 0.0001x the cost**, using only 7M parameters.

---

## 2. How PTRM Works

### The TRM Backbone

TRM performs reasoning by recursively refining a latent state through a small shared network:

```
z_L ← L_level(z_L, input_emb)   # Low-level state (fast)
z_H ← H_level(z_H, z_L)         # High-level state (slow)
logits = lm_head(z_H)
q_halt, q_continue = q_head(z_H[:, 0])
```

The Q-head (`q_halt`, `q_continue`) is trained jointly as a correctness classifier for Adaptive Computation Time (ACT) halting. During standard TRM inference, the Q-head is **not used** — the model runs for a fixed number of steps.

### PTRM Inference Algorithm

PTRM leverages the Q-head at inference time:

```python
# PTRM Inference (Algorithm 1 from paper)
for k in 1..K parallel rollouts:
    Initialize z_H, z_L
    for t in 1..D supervision steps:
        z_H += N(0, σ²I)      # Inject noise
        z_L += N(0, σ²I)      # Inject noise
        z_H, z_L = rec(x, z_H, z_L)  # One TRM step
    y_hat_k = argmax lm_head(z_H)
    q_k = q_continue(z_H[:, 0])      # Q-head score
    
Select k* = argmax_k q_k
Return y_hat_{k*}
```

**Two complementary benefits:**
1. **Escape bad basins** — Noise allows rollouts to explore diverse solution basins; some escape traps where deterministic TRM fails.
2. **Width scaling** — More rollouts (higher K) compound the chance of finding a correct solution. This is a new test-time compute axis (vs. depth scaling).

---

## 3. Quick Start

### Installation

```bash
pip install keras>=3.8 tensorflow>=2.18 numpy
```

### Basic Usage

```python
import keras
import numpy as np
from dl_techniques.models.language.ptrm import (
    PTRM, create_ptrm, PTRMInference, get_ppbench_config
)

# 1. Create a PTRM model (same as TRM architecture)
config = get_ppbench_config()
model = create_ptrm_from_variant(
    variant=config["variant"],
    vocab_size=config["vocab_size"],
    seq_len=config["seq_len"],
    puzzle_emb_len=config["puzzle_emb_len"],
)

# 2. Load pretrained weights (TRM checkpoint works directly)
# model.load_weights("path/to/trm_checkpoint.keras")

# 3. Run PTRM inference
inference = PTRMInference(model, noise_scale=0.2)

batch = {"inputs": np.random.randint(0, 294, size=(4, 100))}

best_logits, best_q, all_q = inference(
    batch,
    num_rollouts=100,
    supervision_steps=48,
)

predictions = keras.ops.argmax(best_logits, axis=-1)
print(f"Selected rollout Q values: {best_q.numpy()}")
```

### Using Preset Configurations

```python
from dl_techniques.models.language.ptrm import PRESET_CONFIGS

# Sudoku-Extreme (5M params, MLP variant)
config = PRESET_CONFIGS["sudoku_extreme"]()
model = create_ptrm_from_variant(**{k: v for k, v in config.items() if k != "inference"})

# PPBench puzzles (7M params, Attention variant)
config = PRESET_CONFIGS["ppbench"]()
model = create_ptrm_from_variant(**{k: v for k, v in config.items() if k != "inference"})

# Maze-Hard
config = PRESET_CONFIGS["maze_hard"]()
model = create_ptrm_from_variant(**{k: v for k, v in config.items() if k != "inference"})

# ARC-AGI-2
config = PRESET_CONFIGS["arc_agi"]()
model = create_ptrm_from_variant(**{k: v for k, v in config.items() if k != "inference"})
```

---

## 4. Component Reference

| Component | Location | Purpose |
| :--- | :--- | :--- |
| **`PTRM`** | `...ptrm.model` | The `keras.Model` (thin TRM subclass). Identical architecture. |
| **`PTRMInference`** | `...ptrm.inference` | Stochastic rollout engine with Q-head selection. |
| **`run_ptrm_inference`** | `...ptrm.inference` | Convenience function for one-off inference. |
| **`create_ptrm`** | `...ptrm.model` | Factory for custom PTRM configurations. |
| **`create_ptrm_from_variant`** | `...ptrm.factory` | Factory from preset variant. |
| **`PRESET_CONFIGS`** | `...ptrm.factory` | Paper-matched configurations. |
| **`TRMReasoningModule`** | `...ptrm.components` | Re-exported TRM reasoning stack. |
| **`TRMInner`** | `...ptrm.components` | Re-exported TRM inner step. |

---

## 5. Configuration & Model Variants

PTRM uses the **exact same architecture variants as TRM**. The paper evaluates two main families:

| Variant | Params | Attention | Use Case |
|---------|--------|-----------|----------|
| `mlp_base` | ~5M | Group-query (MLP-style) | Sudoku-Extreme |
| `att_base` | ~7M | Group-query + RoPE | PPBench, Maze-Hard, ARC-AGI-2 |

### Variant Details

```python
MODEL_VARIANTS = {
    "mlp_tiny":    {"hidden_size": 256, "num_heads": 4,  "h_layers": 2, "l_layers": 2, "halt_max_steps": 8},
    "mlp_small":   {"hidden_size": 384, "num_heads": 6,  "h_layers": 2, "l_layers": 2, "halt_max_steps": 10},
    "mlp_base":    {"hidden_size": 512, "num_heads": 8,  "h_layers": 2, "l_layers": 2, "halt_max_steps": 16},
    "att_small":   {"hidden_size": 384, "num_heads": 6,  "h_layers": 2, "l_layers": 2, "halt_max_steps": 16},
    "att_base":    {"hidden_size": 512, "num_heads": 8,  "h_layers": 2, "l_layers": 2, "halt_max_steps": 16},
    "att_large":   {"hidden_size": 768, "num_heads": 12, "h_layers": 2, "l_layers": 2, "halt_max_steps": 16},
}
```

All variants use: `expansion=2.0`, `attention_type='group_query'`, `ffn_type='swiglu'`, `normalization_type='rms_norm'`.

---

## 6. Inference Parameters

| Parameter | Paper Value (PPBench) | Paper Value (Sudoku) | Paper Value (Maze) | Paper Value (ARC) | Description |
|-----------|----------------------|---------------------|-------------------|------------------|-------------|
| `num_rollouts` (K) | 100 | 100 | 100 | 25 | Parallel stochastic rollouts |
| `supervision_steps` (D) | 48 | 64 | 16 | 16 | Deep recursion steps |
| `noise_scale` (σ) | 0.2 | 0.3 | 1.0 | 0.2 | Gaussian noise std dev |

**Guidance:**
- Higher K → better accuracy, linear compute cost (parallelizable)
- Higher D → more reasoning depth, sequential cost
- σ is task-dependent; paper sweeps σ in Appendix B

---

## 7. Training

PTRM **does not require training** — it uses a pretrained TRM checkpoint directly. To train a TRM backbone for use with PTRM, see the TRM training scripts or the PTRM training pipeline in `src/train/ptrm/`.

The training procedure is identical to TRM:
- Deep supervision with ACT halting
- Loss: `CE(logits, labels) + BCE(q_head, correctness)`
- Uses `dl_techniques.losses.hrm_loss.HRMLoss`

---

## 8. Serialization & Deployment

`PTRM` is fully serializable in the `.keras` format:

```python
model = PTRM(...)
# ... training ...
model.save('my_ptrm_model.keras')

# No custom_objects needed
loaded_model = keras.models.load_model('my_ptrm_model.keras')
assert loaded_model.hidden_size == model.hidden_size
```

The `PTRMInference` class is stateless (holds only the model reference and noise scale) and does not need serialization.

---

## 9. Testing & Validation

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_models/test_ptrm/ -v
```

### Quick Sanity Check

```python
import tensorflow as tf
from dl_techniques.models.language.ptrm import PTRM, PTRMInference

def test_ptrm_inference():
    model = PTRM(vocab_size=100, hidden_size=64, num_heads=4,
                 expansion=2.0, seq_len=50, halt_max_steps=8)
    batch = {"inputs": tf.zeros((4, 50), dtype=tf.int32)}

    # Standard TRM inference (deterministic)
    carry = model.initial_carry(batch)
    for _ in range(8):
        carry, outputs = model(carry, batch, training=False)
    det_logits = outputs["logits"]
    
    # PTRM inference (stochastic)
    inference = PTRMInference(model, noise_scale=0.2)
    best_logits, best_q, _ = inference(batch, num_rollouts=10, supervision_steps=8)
    
    assert best_logits.shape == (4, 50, 100)
    assert best_q.shape == (4,)
```

---

## 10. Troubleshooting & FAQs

- **PTRM is slower than TRM** — Expected. K rollouts multiply inference cost. The paper notes width scaling is more practical than depth scaling because rollouts are parallelizable. For production, vectorize the rollout loop with `tf.vectorized_map`.
- **Q-head doesn't select correct rollout** — On some tasks (Maze-Hard, ARC-AGI-2), the gap between `best-Q@K` and `pass@K` indicates the Q-head is an imperfect verifier. This is a known limitation; future work on stronger verifiers is suggested.
- **Noise scale σ** — If accuracy doesn't improve, try sweeping σ. Paper shows optimal σ varies by task (Appendix B).
- **Using TRM checkpoints** — PTRM is a thin subclass of TRM. Load any TRM `.keras` checkpoint directly into a PTRM instance with matching config.

---

## 11. Citation

```bibtex
@article{sghaier2026probabilistic,
  title={Probabilistic Tiny Recursive Model},
  author={Sghaier, Amin and Parviz, Ali and Jolicoeur-Martineau, Alexia},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}

@article{jolicoeur2025less,
  title={Less is More: Recursive Reasoning with Tiny Networks},
  author={Jolicoeur-Martineau, Alexia},
  journal={arXiv preprint arXiv:2510.04871},
  year={2025}
}
```

---

## 12. License

GPL-3.0 — see repository root `LICENSE`.