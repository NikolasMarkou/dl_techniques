# `dl_techniques.optimization`

Config-driven builders for Keras 3 optimizers, learning-rate schedules and deep-supervision
weights, plus four custom optimizers (`Muon`, `SGLD`, `VSGD`, `Gefen`), a warmup schedule
wrapper, a SLED logits processor and a WeightWatcher-style spectral projection callback.

Everything is built from plain `dict` config, so a whole training recipe can live in JSON.

## Quick start

```python
from dl_techniques.optimization import (
    learning_rate_schedule_builder,
    optimizer_builder,
)

lr_schedule = learning_rate_schedule_builder({
    "type": "cosine_decay",
    "learning_rate": 1e-3,
    "decay_steps": 10_000,
    "warmup_steps": 1_000,
    "warmup_start_lr": 1e-8,
    "alpha": 1e-4,
})

optimizer = optimizer_builder(
    {
        "type": "adamw",
        "weight_decay": 1e-4,
        "gradient_clipping_by_norm": 1.0,          # -> Keras global_clipnorm
        "exclude_from_weight_decay": ["bias", "gamma", "beta"],
    },
    lr_schedule,                                    # 2nd POSITIONAL arg
)

model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy")
```

`lr_schedule(0) == 1e-8`, `lr_schedule(1000) == 1e-3`, then cosine decay from there.

## Entry points

| Import from `dl_techniques.optimization` | Signature | Returns |
|---|---|---|
| `optimizer_builder` | `(config: dict, lr_schedule: float \| LearningRateSchedule)` | `keras.optimizers.Optimizer` |
| `learning_rate_schedule_builder` | `(config: dict)` | `WarmupSchedule` (always wrapped) |
| `create_learning_rate_schedule` | `(initial_lr, schedule_type='cosine', total_epochs=100, warmup_epochs=5, steps_per_epoch=None, warmup_steps=0, warmup_start_lr=1e-8)` | schedule, or a bare `float` for `'constant'` |
| `create_warmup_lr_schedule` | `(learning_rate, num_epochs, steps_per_epoch, warmup_ratio=0.1)` | `WarmupSchedule` over `CosineDecay(alpha=0.0)` |
| `deep_supervision_schedule_builder` | `(config: dict, no_outputs: int, invert_order: bool = False)` | `Callable[[float], np.ndarray]` |
| `WarmupSchedule` | `(warmup_steps, warmup_start_lr=1e-8, primary_schedule=None)` | linear ramp then `primary_schedule(step - warmup_steps)` |
| `Muon`, `SGLD`, `VSGD`, `Gefen` | Keras optimizer classes | see below |
| `WWTailConfig`, `ww_pgd_project`, `WWPGDProjectionCallback` | spectral tail projection | see below |

`sled_builder` / `SledLogitsProcessor` are **not** re-exported; import them from
`dl_techniques.optimization.sled_supervision`.

## `optimizer_builder`

`config["type"]` (lower-cased, stripped) selects the optimizer. Unknown types raise `ValueError`.

| `type` | Class | Type-specific config keys (default) |
|---|---|---|
| `"adam"` | `keras.optimizers.Adam` | `beta_1` (0.9), `beta_2` (0.999), `epsilon` (1e-7), `amsgrad` (False) |
| `"adamw"` | `keras.optimizers.AdamW` | same as Adam, plus `weight_decay` (**0.004**) |
| `"sgd"` | `keras.optimizers.SGD` | `momentum` (0.0), `nesterov` (False) |
| `"rmsprop"` | `keras.optimizers.RMSprop` | `rho` (0.9), `momentum` (0.0), `epsilon` (1e-7), `centered` (False) |
| `"adadelta"` | `keras.optimizers.Adadelta` | `rho` (0.9), `epsilon` (1e-7) |
| `"sgld"` | `SGLD` | `noise_scale` (1.0), `seed` (None) |
| `"vsgd"` | `VSGD` | `ghattg` (30.0), `ps` (1e-8), `tau1` (0.81), `tau2` (0.90), `eps` (1e-8), `weight_decay` (0.0) |
| `"gefen"` | `Gefen` | `beta_1` (0.9), `beta_2` (0.999), `epsilon` (1e-8), `weight_decay` (0.0), `max_block_size` (1024), `min_block_size` (8) |

`Muon` is **not** reachable through `optimizer_builder` — construct it directly.

Keys accepted for every type:

| Key | Effect |
|---|---|
| `gradient_clipping_by_value` | → Keras `clipvalue` |
| `gradient_clipping_by_norm_local` | → Keras `clipnorm` (per-variable L2) |
| `gradient_clipping_by_norm` | → Keras `global_clipnorm` (global L2) |
| `weight_decay` | forwarded when set; otherwise the Keras default applies (`None` everywhere except AdamW's 0.004) |
| `exclude_from_weight_decay` | list of name patterns matched with `re.search`; applied after construction, ignored with a warning on optimizers that lack the method |

Defaults live in `constants.py` (`DEFAULT_*`) — the single source of truth.

### Two traps in `optimizer_builder`

**1. The clipping keys are renamed.** A literal `"clipnorm"` / `"clipvalue"` /
`"global_clipnorm"` key in your config is *silently ignored* — the builder only reads
`gradient_clipping_by_*`. Measured:

```python
optimizer_builder({"type": "adamw", "clipnorm": 1.0}, 1e-3).clipnorm      # -> None
optimizer_builder({"type": "adamw", "gradient_clipping_by_norm_local": 1.0}, 1e-3).clipnorm  # -> 1.0
```

A naive migration from `keras.optimizers.AdamW(clipnorm=1.0)` therefore ships with clipping
disabled and no error. Keras also rejects setting more than one of the three at once, so pick
exactly one `gradient_clipping_by_*` key.

**2. It hard-codes `"name": "AdamW"`** where Keras' own default is `"adamw"`. The name is the
optimizer's variable scope, so slot variables are created under a different path:

```python
optimizer_builder({"type": "adamw"}, 1e-3).name        # -> 'AdamW'
keras.optimizers.AdamW(learning_rate=1e-3).name        # -> 'adamw'
# slot variable path becomes 'AdamW/w_momentum', not 'adamw/w_momentum'
```

That matters for anything keyed on variable paths — optimizer-state checkpoints written by one
construction path will not line up with the other. (Same pattern for `"Adam"`, `"SGD"`,
`"RMSprop"`, `"Adadelta"`, `"SGLD"`, `"VSGD"`; `"gefen"` is already lower-case.)

## Learning-rate schedules

`learning_rate_schedule_builder(config)` takes a **flat** dict and always returns a
`WarmupSchedule`, even at `warmup_steps=0` (a numerical no-op).

| `type` | Required keys | Optional keys (default) |
|---|---|---|
| `"cosine_decay"` | `learning_rate`, `decay_steps` | `alpha` (1e-4) |
| `"exponential_decay"` | `learning_rate`, `decay_steps`, `decay_rate` | — |
| `"cosine_decay_restarts"` | `learning_rate`, `decay_steps` (= first period) | `t_mul` (2.0), `m_mul` (0.9), `alpha` (1e-3) |

Warmup keys, valid for all three: `warmup_steps` (0), `warmup_start_lr` (1e-8). Missing required
keys raise `KeyError`; an unknown `type` raises `ValueError`.

Warmup behaviour:

```
step <  warmup_steps: lr = warmup_start_lr + (target - warmup_start_lr) * step / warmup_steps
step >= warmup_steps: lr = primary_schedule(step - warmup_steps)
```

### The two epoch-facing adapters

`create_learning_rate_schedule` and `create_warmup_lr_schedule` are **deliberately different**
from `schedule_builder` and must not be "unified" with it — dozens of trainers depend on the
current behaviour:

- `create_learning_rate_schedule` returns a **bare** `CosineDecay` when `warmup_steps == 0`,
  hard-codes `alpha=0.01` / `decay_rate=0.9`, and returns a plain `float` for `'constant'`.
- `warmup_epochs` is a **no-op** kept for positional compatibility. Warmup engages only via
  `warmup_steps > 0`, which additionally requires `steps_per_epoch` (else `ValueError`).
- `create_warmup_lr_schedule` sizes warmup as a fraction of the total step budget, always warms
  up, and uses `alpha=0.0`, `warmup_start_lr=1e-7`.

## Deep supervision

`deep_supervision_schedule_builder(config, no_outputs, invert_order=False)` returns a function
of training progress in `[0, 1]` giving one normalised weight per output (weights sum to 1.0).
Config shape is `{"type": ..., "config": {...}}`.

Default ordering: output 0 is the final, highest-resolution head; output `n-1` is the deepest,
lowest-resolution one. `invert_order=True` reverses the returned array.

| `type` | Behaviour | `config` params (default) |
|---|---|---|
| `constant_equal` | uniform weights, constant | — |
| `constant_low_to_high` | fixed tilt toward shallow | — |
| `constant_high_to_low` | fixed tilt toward deep | — |
| `linear_low_to_high` | linear shift deep → shallow | — |
| `non_linear_low_to_high` | same, non-linear ramp | — |
| `custom_sigmoid_low_to_high` | sigmoid transition | `k` (10.0), `x0` (0.5), `transition_point` (0.25) |
| `scale_by_scale_low_to_high` | one scale at a time | — |
| `cosine_annealing` | oscillating emphasis | `frequency` (3.0), `final_ratio` (0.5) |
| `curriculum` | progressively activates outputs | `max_active_outputs` (`no_outputs`), `activation_strategy` (`'linear'` \| `'exp'`) |
| `step_wise` | hard switch at a threshold | `threshold` (0.5) |

```python
from dl_techniques.optimization import deep_supervision_schedule_builder

sched = deep_supervision_schedule_builder({"type": "linear_low_to_high", "config": {}}, 5)
sched(0.0)   # [0.067 0.133 0.2 0.267 0.333]  -> weight on the deep heads
sched(1.0)   # [0.333 0.267 0.2 0.133 0.067]  -> weight on the final head
```

## Custom optimizers

### `Muon`

MomentUm Orthogonalized by Newton-Schulz. Hybrid: rank ≥ 2 non-embedding kernels get the
orthogonalized momentum update, everything else (biases, norm gains, embeddings) is handled by an
integrated auxiliary AdamW. Orthogonalization allows a much larger LR than AdamW.

Use it for Transformers / ConvNets where you want fewer steps to a target loss. Not wired into
`optimizer_builder`.

```python
from dl_techniques.optimization import Muon

optimizer = Muon(
    learning_rate=0.02,      # Muon branch
    momentum=0.95,
    nesterov=True,
    ns_steps=5,              # Newton-Schulz iterations
    adam_learning_rate=1e-3, # auxiliary AdamW branch
    weight_decay=0.0,
    exclude_embedding_names=["embedding", "token_emb", "embed"],
)
```

Routing rule: `rank >= 2` **and** no `exclude_embedding_names` substring in the variable name →
Muon; otherwise AdamW. If Muon is not beating AdamW, check that routing first.

Keller Jordan et al., 2024 — <https://kellerjordan.github.io/posts/muon/>,
<https://github.com/KellerJordan/Muon>

### `SGLD`

SGD plus isotropic Gaussian noise of stddev `sqrt(2 * lr) * noise_scale`. With an LR annealed to
0, the iterates approximate samples from the Bayesian posterior. Use it for posterior sampling /
weight ensembles, or to escape shallow minima. `noise_scale=0.0` is plain SGD.

```python
from dl_techniques.optimization import SGLD

optimizer = SGLD(learning_rate=1e-2, noise_scale=1.0, seed=42, weight_decay=None)
# or: optimizer_builder({"type": "sgld", "noise_scale": 1.0, "seed": 42}, lr_schedule)
```

Snapshot weights periodically after a burn-in to build the posterior ensemble; a single final
checkpoint is a posterior *sample*, not a MAP estimate.

Welling & Teh, ICML 2011, *Bayesian Learning via Stochastic Gradient Langevin Dynamics*.

### `VSGD`

Variational Stochastic Gradient Descent: models the gradient with a probabilistic model and
derives a closed-form adaptive update by stochastic variational inference, keeping per-variable
running statistics. A drop-in adaptive optimizer; the default LR (0.1) is much larger than Adam's.

```python
from dl_techniques.optimization import VSGD

optimizer = VSGD(learning_rate=0.1, ghattg=30.0, tau1=0.81, tau2=0.90)
```

Chen et al., 2024, *VSGD: Variational Stochastic Gradient Descent via Bayesian Online Natural
Gradient*.

### `Gefen`

"Gefen-lite (shared-v)": AdamW with **one second-moment scalar per block** of `period`
contiguous flattened parameters, full-precision momentum unchanged. `period` is the largest
divisor of the variable's element count `<= max_block_size`, falling back to 1 (per-element
AdamW) when no divisor is `>= min_block_size`. Because `period` is a static Python int per
variable, the update is graph-static and `jit_compile` / `model.fit` safe.

Use it when optimizer state is the memory bottleneck and you want an AdamW drop-in.

```python
from dl_techniques.optimization import Gefen

optimizer = Gefen(learning_rate=1e-3, weight_decay=1e-2, max_block_size=1024, min_block_size=8)
```

**Scope caveat:** this is the shared-second-moment part only. There is no uint8 momentum
quantization and no learned codebook, and `period` is chosen from shape rather than gradient
statistics. The momentum buffer is the same size as AdamW's, so this does **not** reproduce the
paper's full optimizer-state reduction — only the second-moment buffer shrinks, from `numel` to
`numel / period` floats per variable.

Inspired by arXiv:2606.13894.

## WW-PGD spectral projection

`ww_pgd_project(model, config, *, epoch, num_epochs, logs=None)` walks the model's rank ≥ 2
non-embedding kernels and, for layers with a large enough power-law tail, reshapes the tail
toward an `r^(-q)` template in place. `WWPGDProjectionCallback(config=..., num_epochs=...,
model=..., csv_path=...)` runs it on the `apply_every_epochs` cadence.

`WWTailConfig` is **off by default** (`enable=False` is a strict no-op). Knobs: `min_tail` (5),
`q` (1.0), `blend_eta` (0.5), `cayley_eta` (0.25), `max_ks_distance` (None), `use_detx` (True),
`warmup_epochs` (0), `ramp_epochs` (5), `apply_every_epochs` (1), `verbose` (False),
`log_layer_stats` (False — turning it on costs one extra SVD per projected layer).

## SLED logits processor

Self Logits Evolution Decoding: contrasts every layer's next-token logits against the final
layer's to improve factuality at generation time.

```python
from dl_techniques.optimization.sled_supervision import sled_builder

processor = sled_builder({
    "type": "sled_v1",
    "config": {
        "evolution_rate": 0.5,      # alpha
        "evolution_scale": 10,      # k, top-k tokens considered
        "temperature": 1.0,         # tau
        "use_tau_in_update": True,
        "inactive_logit_value": -1e9,
    },
})

# all_logits: list of [batch, vocab] tensors, ordered layer 0 -> final layer
final_logits = processor(all_logits)
```

If every layer contrast is misaligned the denominator is zero and the processor logs a warning
and returns the original final-layer logits unchanged.

arXiv:2411.02433.

## Gotchas

- `optimizer_builder`'s second argument is **positional** (`lr_schedule`). There is no
  `learning_rate=` keyword — passing one raises `TypeError`.
- Set at most one of the three `gradient_clipping_by_*` keys; Keras raises if more than one of
  `clipnorm` / `clipvalue` / `global_clipnorm` is set.
- `"adamw"` defaults to `weight_decay=0.004` when the key is absent. Every other type defaults to
  no decay. Do not also add an L2 kernel regularizer — that decays the parameter twice.
- `learning_rate_schedule_builder` requires `decay_steps` in *optimizer steps*, not epochs.
  `create_learning_rate_schedule` takes epochs.
- There is no `"constant"` schedule type in `learning_rate_schedule_builder` — pass a plain float
  as the `lr_schedule` argument instead.
- `Muon` and the WW-PGD tools are not reachable from `optimizer_builder`.

## See also

- `CLAUDE.md` in this directory — module map and authoring rules.
- `train_vision/README.md` — the vision training pipeline built on these builders.
- Tests: `tests/test_optimization/`.
