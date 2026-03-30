# Optimization Package

Optimizer builders, learning rate schedules, and training supervision utilities — all config-driven.

## Public API

```python
from dl_techniques.optimization import (
    optimizer_builder,                    # Creates optimizers (Adam, AdamW, RMSprop, Adadelta)
    learning_rate_schedule_builder,       # Creates LR schedules with warmup
    deep_supervision_schedule_builder,    # Creates deep supervision weight schedules
)
```

## Modules

- `optimizer.py` — `optimizer_builder()`: config-driven optimizer creation with gradient clipping support
- `schedule.py` — `schedule_builder()`: learning rate schedule construction (cosine decay, etc.) with warmup
- `warmup_schedule.py` — Warmup schedule implementation
- `deep_supervision.py` — Deep supervision weight scheduling (linear low-to-high, etc.)
- `sled_supervision.py` — SLED supervision strategy
- `muon_optimizer.py` — `Muon` optimizer: hybrid Muon (Newton-Schulz orthogonalization) for weight matrices + AdamW for other params. 1.35x faster on Transformers
- `sled_supervision.py` — `SledLogitsProcessor`: inference-time logit correction using earlier layer latent knowledge for improved LLM factuality
- `constants.py` — Optimization constants and defaults

### Subpackages
- `train_vision/` — Vision training framework:
  - `framework.py` — End-to-end vision training pipeline

## Conventions

- Config-driven: all builders accept `Dict[str, Any]` configuration
- Flattened config structure for LR schedules (warmup params alongside schedule params)
- Gradient clipping configured via optimizer config (`gradient_clipping_by_norm`, etc.)

## Testing

Tests in `tests/test_optimization/`.
