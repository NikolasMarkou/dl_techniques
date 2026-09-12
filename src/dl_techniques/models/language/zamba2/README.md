# Zamba2: Hybrid Mamba2 + Shared-Attention SSM/Transformer LM

A Keras 3 implementation of Zyphra's **Zamba2** architecture: a causal language model that
interleaves per-depth Mamba2 state-space blocks with a small number of **shared** attention+MLP
"mem-blocks" reused at several depths, where each reuse ("occurrence") of the shared MLP block's
up-projection carries its own small additive LoRA delta — letting one physical block specialize
per depth without multiplying its parameter count.

> **No pretrained weights are distributed by this library.** `create_zamba2(pretrained=True)` /
> `Zamba2Model(pretrained=True)` raise `NotImplementedError` — no Zamba2 checkpoint is fetched
> from Hugging Face or converted here. Warm-start a built model via
> `model.load_weights("/path/to/weights.keras")` instead.

---

## 1. Overview

Zamba2's central idea is **parameter sharing across depth**, applied selectively: the SSM
(Mamba2) mixer blocks are *not* shared — every depth position gets its own, independently
weighted `Zamba2MambaBlock` (`decisions.md` D-005) — but a handful of attention+MLP "mem-blocks"
*are* shared, invoked at multiple depths with identical base weights. To let a shared block
specialize per depth anyway, each invocation ("occurrence") of the shared MLP block's
up-projection gets its own low-rank LoRA delta, selected by a monotonically increasing global
occurrence counter that is independent of which physical mem-block slot is invoked.

This package reuses `dl_techniques.models.language.mamba.components_v2.Mamba2ResidualBlock`
directly for the SSM mixer (wrapped by `Zamba2MambaBlock`, a thin kwarg-translation adapter), and
the repo's existing `RMSNorm`, `RotaryPositionEmbedding` and `'multi_head'` attention factory
entry for the shared attention block. The only genuinely new primitive is `LoRAAdapter`
(`decisions.md` D-001) — no additive, frozen-base low-rank adapter existed anywhere in the repo
before this package.

---

## 2. Architecture

```
input_ids [B, S]
       │
       ▼ Embedding(vocab_size, hidden_size)
original_embedding [B, S, H]  ──────────────────────┐
       │                                             │  (re-read at
       ▼                                             │   every 'g')
hidden_state = original_embedding                    │
       │                                             │
       ▼  for token in layer_mapping:                │
  ┌────┴─────────────────────────────────┐           │
  │ 'm': hidden_state =                  │           │
  │   Zamba2MambaBlock_i(hidden_state)   │  (fresh instance per 'm')
  │ 'g': hidden_state =                  │           │
  │   Zamba2SharedAttentionBlock_{i%N}(  │◄──────────┘
  │     hidden_state, original_embedding)│  (one of N=num_mem_blocks
  │   hidden_state =                     │   shared instances,
  │   Zamba2SharedMLPBlock_{i%N}(         │   round-robin; occurrence
  │     hidden_state,                     │   counter increments once
  │     occurrence_idx=<g counter>)       │   per 'g', independent of
  └────┬─────────────────────────────────┘   which physical slot)
       ▼
final RMSNorm
       │
       ▼
logits = hidden_state @ embedding.weights[0]^T   (tied LM head)
[B, S, vocab_size]
```

### 2.1 `Zamba2MambaBlock` (`'m'` positions)

A thin adapter around `Mamba2ResidualBlock`, translating Zamba2's constructor vocabulary
(`d_model`, `d_state`, `d_conv`, `expand`, `headdim`) onto the shared primitive's own kwargs, plus
pass-through for `d_ssm`, `ngroups`, `norm_epsilon`, `rmsnorm`, `norm_before_gate`, the `dt_*`
initialization knobs, and bias flags. Every `'m'` position builds its own instance — never shared.

### 2.2 `Zamba2SharedAttentionBlock` (half of every `'g'` position)

`RMSNorm(hidden_state) -> concat([original_embedding, normed]) -> Dense(d_model) ->`
`RotaryPositionEmbedding -> causal 'multi_head' attention -> output Dense ->` residual add onto
the **original** `hidden_state` (not the concatenated tensor). No LoRA on this block — scope
decision `decisions.md` D-001. One physical instance is built per `num_mem_blocks` slot and reused
round-robin across however many `'g'` entries `layer_mapping` contains.

### 2.3 `Zamba2SharedMLPBlock` (the other half of every `'g'` position)

`RMSNorm(hidden_state) ->` parallel `gate_proj`/`up_proj` Dense `->` an owned `LoRAAdapter` adds
its delta onto `up_proj`'s output, selected by the caller's `occurrence_idx` `-> SiLU(gate) * up`
`-> down_proj Dense -> ` residual add. LoRA is scoped to `up_proj` only. `hidden_dim` derives via
the PaLM-style 2/3-rule-then-round-up arithmetic (matching `SwiGLUFFN`) when not given explicitly.

### 2.4 `LoRAAdapter`

Owns `A: (num_occurrences, input_dim, rank)` and `B: (num_occurrences, rank, output_dim)`
(`B` zero-initialized, so the delta is exactly zero at construction — non-vacuous once trained).
`call(x, occurrence_idx)` returns `(x @ A[occurrence_idx]) @ B[occurrence_idx] * (alpha / rank)`,
a pure additive delta the caller is responsible for summing onto its own base projection.

---

## 3. Quick Start

```python
import keras
from dl_techniques.models.language.zamba2 import create_zamba2

model = create_zamba2("zamba2_mini")
input_ids = keras.random.randint((2, 32), 0, 100277, dtype="int32")
logits = model(input_ids)
print(logits.shape)  # (2, 32, 100277)
```

---

## 4. Variants

`Zamba2Model.MODEL_VARIANTS` (repo-scale `mini`/`small`/`base`, **not** the Zamba2 paper's own
2.7B/7B table — an explicit non-goal, `decisions.md` D-002; each entry also carries a
`"description"`, which `from_variant` pops before construction). `layer_mapping` is **derived**
from `num_mamba_blocks`/`num_mem_blocks` at `from_variant` time (mostly `'m'`, with `'g'` inserted
at evenly-spaced intervals — scaled down from the original config's roughly-every-6th-position
density), never stored literally, so the two counts cannot drift out of sync with the mapping's
actual composition.

| Variant | `hidden_size` | `num_mamba_blocks` | `num_mem_blocks` | `num_heads` | `max_seq_len` | `lora_rank` |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `zamba2_mini` | 256 | 8 | 2 | 4 | 512 | 4 |
| `zamba2_small` | 512 | 18 | 6 | 8 | 1024 | 8 |
| `zamba2_base` | 768 | 24 | 8 | 12 | 2048 | 16 |

Re-derive this table with `for name, cfg in Zamba2Model.MODEL_VARIANTS.items(): print(name, cfg)`
rather than trusting it. `vocab_size` defaults to `100277` (Tiktoken `cl100k_base`) for every
variant and is not itself a variant-distinguishing field; override it with
`create_zamba2("zamba2_mini", vocab_size=...)`.

```python
from dl_techniques.models.language.zamba2 import Zamba2Model, create_zamba2

model = Zamba2Model.from_variant("zamba2_small")             # named variant
model = Zamba2Model.from_variant("zamba2_mini", lora_rank=8) # variant + override
model = create_zamba2("zamba2_base")                          # factory, same delegation
```

---

## 5. The Pretrained-Weights Contract

```python
from dl_techniques.models.language.zamba2 import create_zamba2

try:
    create_zamba2("zamba2_mini", pretrained=True)
except NotImplementedError as e:
    print(f"NotImplementedError: {e}")
```

No Zamba2 checkpoint is distributed with this repository; `pretrained=True` raises rather than
silently returning a random-init model (`decisions.md` D-004). Warm-start a built model with
`model.load_weights("/path/to/weights.keras")` instead.

---

## 6. Serialization

```python
import keras
import numpy as np
from dl_techniques.models.language.zamba2 import create_zamba2

model = create_zamba2("zamba2_mini")
ids = np.random.randint(0, 100277, size=(1, 16)).astype("int32")
before = model(ids)

model.save("zamba2_demo.keras")
restored = keras.models.load_model("zamba2_demo.keras")  # no custom_objects needed
after = restored(ids)

print(float(np.max(np.abs(np.array(before) - np.array(after)))))  # 0.0
```

`get_config()` round-trips every constructor argument, including `layer_mapping`,
`num_mem_blocks`, and every Mamba2/attention/LoRA hyperparameter.

---

## 7. Training

`src/train/zamba2/` (added in a later plan step) wires Pattern-3 (`train.common.nlp`) causal-LM
pre-training against the repo's Wikipedia corpus and a Tiktoken `cl100k_base` tokenizer —
substituting for the paper's own Zyda corpus / Mistral tokenizer, which have no infrastructure in
this repo (`decisions.md` D-003).

---

## 8. Testing

```bash
CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_models/test_zamba2/ -q
```

The suite covers initialization and validation errors, forward shapes, gradient flow to every
trainable weight (including every exercised LoRA `A`/`B` pair), `get_config`/`.keras` round trips,
every named variant, the `pretrained=True` contract, and three decisive guards:

- `test_shared_weights_identical_across_depth.py` — a shared mem-block's weights are `is`-identical
  (not just equal-valued) across two different depth positions.
- `test_lora_differs_per_occurrence.py` — two calls to the same shared MLP block with different
  `occurrence_idx` values produce different outputs after training (and identical outputs with the
  *same* `occurrence_idx`) — the plan's single highest-risk claim.
- `test_the_causal_mask_is_honoured.py` — a three-armed future-leak probe on every attention block
  in the stack, at bit-exact `atol=0`.

---

## 9. Citation

```bibtex
@article{glorioso2024zamba2,
  title={Zamba2: A Compact and Fast Hybrid Model},
  author={Glorioso, Paolo and Anthony, Quentin and Tokpanov, Yury and Whittington, James and
          Pilault, Jonathan and Ibrahim, Adam and Millidge, Beren},
  journal={Zyphra Technical Report},
  year={2024},
  url={https://arxiv.org/abs/2411.15242}
}

@article{glorioso2024zamba,
  title={Zamba: A Compact 7B SSM Hybrid Model},
  author={Glorioso, Paolo and Anthony, Quentin and Tokpanov, Yury and Golubeva, Anna and
          Shyam, Vasudev and Whittington, James and Pilault, Jonathan and Millidge, Beren},
  journal={arXiv preprint arXiv:2405.16712},
  year={2024}
}
```
