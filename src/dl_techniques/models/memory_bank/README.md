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
| `MemoryReadController` | Layer | Multi-query top-K STE retrieval + gated residual injection. Holds 5 anti-collapse aux losses. Two gates apply: the `enable_*` flags decide which terms exist in the graph (static, read at trace time), and the `aux_scale` tensor argument scales them at runtime — that is the one the curriculum drives. **MQA**: K and V are shared across heads; only Q is per-head. **Gate**: per-channel (`D→D`) sigmoid with bias init `−3.0`. **Positional-7 `call`**: incompatible with the Keras Functional API — use a subclassed model. |
| `PhaseScheduler` | Callback | Reads `model._global_step`, computes the current phase, assigns it to the `model.current_phase` **Variable**, zeroes/restores the backbone optimizer's learning-rate Variable, runs KMeans warmup at P1→P2. It flips **no** Python attribute — see "Why the curriculum is all Variables". |
| `WaveFieldMemoryLLM` | Model | Dual-tap LM. Custom `compile` accepts `backbone_optimizer` + `memory_optimizer`; custom `train_step` splits grads by `Variable.path` prefix and masks the backbone's by the phase gate. |

## Variable naming convention (load-bearing)

Trainable variables route to the **memory** optimizer iff the **leading component of `Variable.path`**, or the bare `Variable.name`, starts with `memory_` or `gate_`; everything else routes to the **backbone** optimizer. Under Keras 3 `.name` is the bare weight name (`kernel`/`bias`/`gamma`) and only `.path` carries the layer hierarchy, so a `.name`-only rule sees nothing but direct `add_weight` calls. Layers are explicitly named with these prefixes:

- `memory_lt_bank`, `memory_wm_bank`, `memory_wm_W_K`, `memory_wm_W_V`
- `memory_write_controller`, `memory_read_controller`
- `memory_read_W_Q`, `memory_read_W_out`, `memory_read_out_norm`, `memory_read_log_temp`, `memory_read_log_temp_nce`
- `gate_W_g`
- `memory_K_lt`, `memory_V_lt`, `memory_current_phase`, `memory_global_step`

Routing uses a **leading-component prefix match** on `path.split('/')[0]` (plus the bare `.name`) — not a substring match — so the backbone's per-block `gate_proj` Dense is not captured. Renaming any of the above silently misroutes gradients. Adding new memory weights requires a `memory_` prefix on the leading path component (or on the weight's own name, for weights the model creates on itself).

## Curriculum

```
PHASE_WARMUP            = 1   [0, P1)              backbone trains, memory bypassed, aux losses scaled to 0
                                                   (injection AND aux terms zeroed via `current_phase != PHASE_WARMUP`)
PHASE_FREEZE_BACKBONE   = 2   [P1, P1+P2)          backbone FROZEN, memory trains, aux losses ON
                                                   KMeans warmup runs once at boundary (seeds K_lt — W_Q-projected)
PHASE_FULL              = 3   [P1+P2, P1+P2+P3)    backbone trains again, memory trains, aux losses ON
PHASE_EXTEND            = 4   ≥ sum                same surface as PHASE_FULL (no-op extension)
```

The four phase constants are exported from `phase_scheduler.py` (D2). Use them instead of bare integer literals.

Defaults: P1 = 50k, P2 = 25k, P3 = 100k. `--init-from` in the training script sets `phase1_steps=0` to skip Phase 1 when warm-starting from a `WaveFieldLLM` checkpoint.

`current_phase` and `_global_step` live as `add_weight(trainable=False)` so they survive `model.save` / `load_model` round-trips.

### Why the curriculum is all Variables

`fit()` traces `train_function` **before** `on_train_begin` (`keras/src/backend/tensorflow/trainer.py:360-361`) and rebuilds it only from `compile()` (`keras/src/trainers/trainer.py:187`). A callback that flips a Python attribute at a phase boundary therefore changes nothing the graph will ever read again. Until 2026-08-15 this scheduler set `layer.trainable = False` and `read_controller.enable_* = True`, so at the documented 50k-step boundary the log said "entered phase 2" while the backbone kept training and not one anti-collapse loss ever contributed a gradient (C-28).

Everything the phase implies is now a multiplicative gate read from `current_phase` inside the traced graph:

| Phase effect | Mechanism | Where |
|---|---|---|
| memory injection on/off | `injection * (current_phase != PHASE_WARMUP)` | `WaveFieldMemoryLLM.call` |
| aux losses on/off | same tensor, passed as `aux_scale` and multiplied into every `add_loss` term | `MemoryReadController._maybe_add_aux_losses` |
| backbone freeze (gradient) | `grad * (current_phase != PHASE_FREEZE_BACKBONE)` on the backbone partition | `WaveFieldMemoryLLM.train_step` |
| backbone freeze (optimizer state) | backbone optimizer's learning-rate Variable assigned `0.0` | `WaveFieldMemoryLLM.set_backbone_optimizer_active` |

The last row is not redundant: with Adam/AdamW a gradient of exactly zero still moves a weight, from phase-1 moment estimates and from decoupled weight decay (`variable -= variable * wd * lr`, which no gradient mask can reach). Both terms scale with the learning rate, so zeroing it makes the freeze exact — measured at `1e-4` of drift per step otherwise (`lr=1e-2`, `weight_decay=0.01`). The cost of the whole design is that phase 1 computes aux losses it multiplies by zero and phase 2 computes backbone gradients it multiplies by zero: a backward pass paid for parameters that do not update, in exchange for a curriculum that reaches the graph at all.

Because the gradient half reads the *saved* `current_phase`, a model reloaded mid-curriculum freezes correctly even with no `PhaseScheduler` attached; the optimizer-state half needs the callback (or a manual `set_backbone_optimizer_active(False)`).

A `PhaseScheduler` used on a backbone optimizer built with a `LearningRateSchedule` raises: a schedule's rate is not an assignable Variable, so the freeze could not be honoured.

## Anti-collapse aux losses

Computed only when `training=True`, and only for terms whose `enable_*` flag was set **at construction** (`WaveFieldMemoryLLM(enable_aux_losses=True)`, the default, sets the five below). Every emitted term is then multiplied by the runtime `aux_scale` gate, which is `0.0` in Phase 1 and `1.0` from Phase 2 on. Do not flip an `enable_*` flag from a callback — it decides graph shape and is read once, at trace time.

| Loss | Effect | Default λ |
|---|---|---|
| `gate_entropy` | Maximizes Bernoulli entropy of `g` (push gate away from 0/1 saturation). | 1e-3 |
| `load_balance` | Switch-Transformer-style: `λ · S_lt · Σᵢ stop_grad(fᵢ) · pᵢ` over `M_LT` slice. | 1e-2 |
| `z_loss` | `λ · mean(logsumexp(sim_lt)²)` (Mesh-TF style). | 1e-3 |
| `diversity` | `λ · mean(off-diag (cos K_lt)²)` over a `diversity_subsample` random subsample. | 1e-3 |
| `infonce` | **B2 redesigned (D-001):** anchor = q_emb (router-mixed V_lt mean over heads); positive = top-1 hard-routed V_lt row mean over heads with stop_gradient; negs = `infonce_negatives` random V_lt rows. Cosine in d_v space, learnable τ = `softplus(memory_read_log_temp_nce) + 1e-3`. | 5e-3 |
| `v_diversity` (O6, opt-in) | Mirrors `diversity` but over V_lt. Default off (`enable_v_diversity=False`). | 1e-3 |

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
from dl_techniques.models.memory_bank import memory_llm_custom_objects  # O9: re-exported from package __init__
m = keras.models.load_model("model.keras", custom_objects=memory_llm_custom_objects())
```

### Opt-in features

| Feature | Constructor flag | Default | Notes |
|---|---|---|---|
| Per-head K/V (full MHA over the bank) | `multi_head_keys=True` | `False` | O4. Allocates `(S_lt, num_heads, d_*)` for K_lt/V_lt and reshapes WM K/V. |
| V_lt diversity aux loss | `enable_v_diversity=True` | `False` | O6. Mirrors K_lt diversity. |
| Anti-collapse aux losses | `enable_aux_losses=False` | `True` | C-28. Emits the five aux terms into the graph, where the phase gate scales them. `False` removes them entirely and no curriculum can bring them back. |
| top_k schedule | `top_k_schedule=callable` | `None` | O7. Applied by `PhaseScheduler` on phase transitions. NOT serialized. Use `linear_top_k_anneal(start, end, end_step)` helper. |
| Reset memory | `model.reset_memory(seed=None)` | — | O3. Re-randomizes K_lt/V_lt, zeros phase + step. |
| Periodic stats | Pass `MemoryStats(...)` callback to `fit` | — | O2. Logs top-K hits, gate percentiles, key utilization, V_lt effective rank every `log_every` batches. |

### Serving / incremental decoding

**Not supported.** The backbone is FFT-based and recomputes the full sequence per call (DECISION plan_2026-05-09_0f39a086/D-002). The memory-bank read/write controllers themselves are incremental-friendly, but a streaming variant of the wave-field attention (or a swap to MHA decoder for serving) is required first. See `WaveFieldMemoryLLM` class docstring for the two future paths.

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
memory_banks.py          # LongTermMemoryBank, WorkingMemoryBank (multi_head_keys aware)
write_controller.py      # MemoryWriteController (T<=max_seq_len assert)
read_controller.py       # MemoryReadController + 6 aux losses (gate_entropy, load_balance, z_loss, K_lt diversity, V_lt diversity, redesigned InfoNCE)
phase_scheduler.py       # PhaseScheduler callback + PHASE_* constants + top_k schedule
memory_stats.py          # MemoryStats diagnostic callback (O2)
wave_field_memory_llm.py # WaveFieldMemoryLLM, split_trainable_by_prefix, linear_top_k_anneal, memory_llm_custom_objects
__init__.py              # Re-exports memory_llm_custom_objects, MemoryStats
```

Trainer: **none ships with this package.** `src/train/wave_field/train_memory.py` used to
train it and was DELETED on user instruction on 2026-08-13 (last present at commit
`9f3208319`); nothing replaced it. The model package itself is untouched and fully
tested — write your own trainer, or recover the old one from git history, but do not
assume a runnable `python -m train.…` entry point exists for `WaveFieldMemoryLLM`.

Tests: `tests/test_models/test_memory_bank/`.
