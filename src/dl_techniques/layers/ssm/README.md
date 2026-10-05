# `dl_techniques.layers.ssm`

Selective state space (SSM) layers with linear-time sequence modeling.

## Layers

| Key | Class | What it does |
|---|---|---|
| `selective_ssm` | `SelectiveSSMLayer` | Generic S6 scan (Mamba v1): causal depthwise conv + input-dependent discretization, `(B, L, D)` in/out |
| `context_mamba` | `ContextMambaLayer` | MambaLCT temporal wrapper: `(B, T, L, D)` frames + `(B, Nc, D)` context in, enhanced frames + updated context out |

## Factory

```python
from dl_techniques.layers.ssm import create_ssm_layer

ssm = create_ssm_layer("selective_ssm", d_model=512)
ctx = create_ssm_layer("context_mamba", d_model=512)
```

`create_ssm_layer` raises `ValueError` on any undeclared keyword. The
registry key set and each entry's `required_params` / `optional_params`
are public API pinned by `tests/test_layers/test_factory_registry_drift.py`.

## Notes

- The token-sequence Mamba foundation models stay in
  `models/language/mamba/`; this package holds the modality-agnostic
  primitives they and `models/vision/mambalct/` (planned) share.
- Scan runs under `keras.ops.while_loop` in the variable dtype; half
  precision compute dtype does not make it faster but stays correct.
