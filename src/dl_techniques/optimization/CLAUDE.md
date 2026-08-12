# Optimization Package

Optimizer builders, learning rate schedules, and training supervision utilities — all config-driven.

## Public API

```python
from dl_techniques.optimization import (
    optimizer_builder,                    # Creates optimizers (Adam, AdamW, SGD, RMSprop, Adadelta, Muon, SGLD, VSGD, Gefen)
    learning_rate_schedule_builder,       # Creates LR schedules with warmup (config-driven)
    create_learning_rate_schedule,        # Epoch-facing: 'cosine' | 'exponential' | 'constant'
    create_warmup_lr_schedule,            # Epoch-facing: warmup RATIO + cosine (NLP default)
    WarmupSchedule,                       # Linear warmup wrapper around any primary schedule
    deep_supervision_schedule_builder,    # Creates deep supervision weight schedules
    Muon,                                 # Newton-Schulz orthogonalization optimizer
    SGLD,                                 # Stochastic Gradient Langevin Dynamics
    VSGD,                                 # Variational SGD — SVI-derived adaptive optimizer (Chen et al. 2024)
    Gefen,                                # Gefen-lite (shared-v) — memory-lean AdamW with block-shared second moment
)
```

## Modules

- `optimizer.py` — `optimizer_builder()`: config-driven optimizer creation with gradient clipping and `exclude_from_weight_decay` support
- `schedule.py` — `schedule_builder()`: learning rate schedule construction (cosine decay, etc.) with warmup. Also holds the two epoch-facing adapters `create_learning_rate_schedule()` and `create_warmup_lr_schedule()` (moved here from `train.common`, which re-exports them)
- `warmup_schedule.py` — Warmup schedule implementation

> **Two schedule entry points, deliberately different.** `schedule_builder` ALWAYS wraps its result in a `WarmupSchedule` (a zero-step warmup is a numerical no-op); `create_learning_rate_schedule` returns a BARE `CosineDecay` when `warmup_steps=0`, hard-codes `alpha`/`decay_rate`, and returns a plain float for `'constant'`. Do not "unify" them — dozens of trainers depend on the current behaviour. See the comment above their definitions in `schedule.py`.
>
> `optimizer.py` used to define its own `learning_rate_schedule_builder` and `ScheduleType`. Those were unused duplicates that diverged from `schedule.py` (bare vs wrapped at `warmup_steps=0`) and were only ever exercised by the test suite. They have been deleted; `schedule.py` is the single source of truth.
- `deep_supervision.py` — Deep supervision weight scheduling (linear low-to-high, etc.)
- `sled_supervision.py` — SLED supervision strategy
- `muon_optimizer.py` — `Muon` optimizer: hybrid Muon (Newton-Schulz orthogonalization) for weight matrices + AdamW for other params. 1.35x faster on Transformers
- `sgld_optimizer.py` — `SGLD`: Stochastic Gradient Langevin Dynamics. SGD with calibrated Gaussian noise injected into the update for Bayesian / posterior-sampling training. Wired into `optimizer_builder` (type `"sgld"`)
- `vsgd_optimizer.py` — `VSGD`: Variational Stochastic Gradient Descent (Chen et al. 2024). SVI-derived adaptive optimizer maintaining per-variable running statistics (`mug`, `bg`, `bhg`). Wired into `optimizer_builder` (type `"vsgd"`)
- `gefen_optimizer.py` — `Gefen`: Gefen-lite (shared-v), a memory-lean AdamW variant (arxiv 2606.13894) that shares one second-moment estimate per block of `period` shape-derived parameters while keeping full-precision momentum; `jit_compile`/`model.fit`-safe drop-in for AdamW. Compresses second-moment state only (no momentum quantization / codebook). Wired into `optimizer_builder` (type `"gefen"`)
- `constants.py` — Optimization constants and defaults

### Subpackages
- `train_vision/` — Vision training framework:
  - `framework.py` — End-to-end vision training pipeline

## Conventions

- Config-driven: all builders accept `Dict[str, Any]` configuration
- Flattened config structure for LR schedules (warmup params alongside schedule params)
- Gradient clipping configured via optimizer config (`gradient_clipping_by_norm`, etc.)
- Weight-decay exclusions via `exclude_from_weight_decay: List[str]` on the optimizer config (name patterns matched with `re.search`; the standard recipe is `["bias", "gamma", "beta"]`). Prefer this over calling `optimizer.exclude_from_weight_decay(...)` by hand after construction — the factory applies it before the optimizer is built, which is the only point Keras accepts it.
- `constants.py` is the single source of truth for `DEFAULT_*` values; do not re-declare them in a builder module.

## Testing

Tests in `tests/test_optimization/`.
