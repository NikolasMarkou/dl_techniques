# memory_bank

Dual-tap neural memory module for `WaveFieldLLM`. Adds a persistent long-term key/value bank plus a per-batch working-memory bank, queried via top-K straight-through-estimator (STE) retrieval and a gated residual injection. Trained under a 4-phase curriculum with a dual-optimizer custom `train_step`.

## Design

```
                  pre-block hidden (X_W)            pre-block hidden (X_R)
                          │                                │
   block_0 ─ … ─ block_{L_write-1} ─ block_{L_write} ─ … ─ block_{L_read-1} ─ block_{L_read} ─ … ─ block_{depth-1} ─ final_norm ─ lm_head
                          │                                │
                          ▼                                ▼
                MemoryWriteController          MemoryReadController
                  ├─ WorkingMemoryBank            ├─ Q = X_R · W_Q  (multi-query: H heads share K/V)
                  └─ right-pad to max_seq_len     ├─ K_total = [K_lt | K_wm], V_total = [V_lt | V_wm]
                                                  ├─ sim = Q·Kᵀ/√d_k + causal_mask + padding_mask
                                                  ├─ soft_w = softmax(sim/τ),  τ = softplus(log_temp)+0.1
                                                  ├─ top_k indices → hard_w (renormalized soft on top-K)
                                                  ├─ STE: routing = soft_w + sg(hard_w − soft_w)
                                                  ├─ V = einsum(routing, V_total)
                                                  ├─ V_proj = LN(W_out·concat_heads + b_out)
                                                  ├─ g = σ(X_R · W_g + b_g),  b_g init = −3.0
                                                  └─ injection = g · V_proj   (caller adds residual)
```

Tap depths: `L_write = max(1, depth // 3)`, `L_read = max(L_write+1, 2*depth // 3)`. Constraint: `L_write < L_read < depth`.

`M_static = s_lt + max_seq_len` is fixed at `__init__` so `ops.one_hot(num_classes=M_static)` traces with a static last-axis under XLA. Working-memory keys/values are right-padded to `max_seq_len`; a padding mask carries 1.0 on real positions.

## Components

| Component | Kind | Purpose |
|---|---|---|
| `LongTermMemoryBank` | Layer | Persistent `(K_lt, V_lt)` of `S_lt` slots. KMeans-seeded at the Phase 1→2 boundary via `assign_keys_from_kmeans`. |
| `WorkingMemoryBank` | Layer | Stateless `(B,T,D) → (K_wm, V_wm)` projector. K has no bias (PE-free key direction); V has bias. |
| `MemoryWriteController` | Layer | Wraps `WorkingMemoryBank` and right-pads to `max_seq_len`. Returns `(K_wm, V_wm, padding_mask)`. |
| `MemoryReadController` | Layer | Multi-query top-K STE retrieval + gated residual injection. Holds 5 anti-collapse aux losses, each gated by an `enable_*` flag. **MQA**: K and V are shared across heads; only Q is per-head. **Gate**: per-channel (`D→D`) sigmoid with bias init `−3.0`. **Positional-7 `call`**: incompatible with the Keras Functional API — use a subclassed model. |
| `PhaseScheduler` | Callback | Reads `model._global_step`, computes the current phase, flips `trainable` flags + aux flags, runs KMeans warmup at P1→P2. |
| `WaveFieldMemoryLLM` | Model | Dual-tap LM. Custom `compile` accepts `backbone_optimizer` + `memory_optimizer`; custom `train_step` splits grads by variable-name prefix. |

## Variable naming convention (load-bearing)

Trainable variables route to the **memory** optimizer iff their `.name` contains `memory_` or `gate_`; everything else routes to the **backbone** optimizer. Layers are explicitly named with these prefixes:

- `memory_lt_bank`, `memory_wm_bank`, `memory_wm_W_K`, `memory_wm_W_V`
- `memory_write_controller`, `memory_read_controller`
- `memory_read_W_Q`, `memory_read_W_out`, `memory_read_out_norm`, `memory_read_log_temp`
- `gate_W_g`
- `memory_K_lt`, `memory_V_lt`, `memory_current_phase`, `memory_global_step`

Renaming any of these silently misroutes gradients. Adding new memory weights requires a `memory_` prefix.

## Curriculum

```
Phase 1: [0, P1)              backbone trainable, memory trainable, aux losses OFF, memory bypassed
                              (gate output zeroed via `current_phase == 1` check in forward)
Phase 2: [P1, P1+P2)          backbone FROZEN, memory trainable, aux losses ON
                              KMeans warmup runs once at boundary (seeds K_lt)
Phase 3: [P1+P2, P1+P2+P3)    backbone unfrozen, memory trainable, aux losses ON
Phase 4: ≥ sum                same trainable surface as Phase 3 (no-op extension)
```

Defaults: P1 = 50k, P2 = 25k, P3 = 100k. `--init-from` in the training script sets `phase1_steps=0` to skip Phase 1 when warm-starting from a `WaveFieldLLM` checkpoint.

`current_phase` and `_global_step` live as `add_weight(trainable=False)` so they survive `model.save` / `load_model` round-trips.

## Anti-collapse aux losses

All gated by per-flag enable booleans. Computed only when `training=True`.

| Loss | Effect | Default λ |
|---|---|---|
| `gate_entropy` | Maximizes Bernoulli entropy of `g` (push gate away from 0/1 saturation). | 1e-3 |
| `load_balance` | Switch-Transformer-style: `λ · S_lt · Σᵢ stop_grad(fᵢ) · pᵢ` over `M_LT` slice. | 1e-2 |
| `z_loss` | `λ · mean(logsumexp(sim_lt)²)` (Mesh-TF style). | 1e-3 |
| `diversity` | `λ · mean(off-diag (cos K_lt)²)` over a `diversity_subsample` random subsample. | 1e-3 |
| `infonce` | Contrast routed-V mean against `infonce_negatives` random V_lt rows. **See bug audit.** | 5e-3 |

## Usage

```python
from dl_techniques.models.memory_bank.wave_field_memory_llm import WaveFieldMemoryLLM
from dl_techniques.models.memory_bank.phase_scheduler import PhaseScheduler

model = WaveFieldMemoryLLM.from_variant("small")  # tiny|small|medium|large|xl

backbone_opt = keras.optimizers.AdamW(learning_rate=3e-4)
memory_opt   = keras.optimizers.AdamW(learning_rate=1e-3)
model.compile(
    backbone_optimizer=backbone_opt,
    memory_optimizer=memory_opt,
    loss=MaskedCausalLMLoss(...),
)

scheduler = PhaseScheduler(
    phase1_steps=50_000, phase2_steps=25_000, phase3_steps=100_000,
    warmup_dataset=train_ds.take(64), warmup_num_batches=64,
)
model.fit(train_ds, validation_data=val_ds, callbacks=[scheduler], epochs=N)
```

To deserialize:

```python
from dl_techniques.models.memory_bank.wave_field_memory_llm import memory_llm_custom_objects
m = keras.models.load_model("model.keras", custom_objects=memory_llm_custom_objects())
```

## Variants

| Variant | embed_dim | depth | heads | max_seq_len | d_k | d_v | s_lt |
|---|---|---|---|---|---|---|---|
| tiny   |  256 |  4 |  4 |  512 |  64 | 128 |  4 096 |
| small  |  768 | 12 | 12 | 1024 | 128 | 256 | 16 384 |
| medium | 1024 | 24 | 16 | 1024 | 128 | 512 | 32 768 |
| large  | 1280 | 36 | 20 | 1024 | 128 | 512 | 65 536 |
| xl     | 1600 | 48 | 25 | 1024 | 128 | 512 | 65 536 |

Constraints: `d_k ≠ d_v`, `d_v < embed_dim`, `embed_dim % num_heads == 0`, `top_k ∈ (0, s_lt + max_seq_len]`.

## Files

```
memory_banks.py          # LongTermMemoryBank, WorkingMemoryBank
write_controller.py      # MemoryWriteController
read_controller.py       # MemoryReadController + 5 aux losses
phase_scheduler.py       # PhaseScheduler callback
wave_field_memory_llm.py # WaveFieldMemoryLLM, split_trainable_by_prefix, memory_llm_custom_objects
```

Trainer: `src/train/wave_field_llm/train_memory.py`. Tests: `tests/test_models/test_memory_bank/`.
