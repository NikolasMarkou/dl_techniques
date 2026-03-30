# Regularizers Package

Advanced Keras regularizers beyond standard L1/L2 — targeting weight structure, sparsity, and orthogonality.

## Public API

```python
from dl_techniques.regularizers import (
    BinaryPreferenceRegularizer,          # Pushes weights toward {0, 1}
    TriStatePreferenceRegularizer,        # Pushes weights toward {-1, 0, 1}
    EntropyRegularizer,                   # Targets specific Shannon entropy level
    SoftOrthogonalConstraintRegularizer,  # Encourages column orthogonality
    SoftOrthonormalConstraintRegularizer, # Orthogonality + unit norm
    SRIPRegularizer,                      # Spectral Restricted Isometry Property
    # Factory functions
    create_binary_preference_regularizer,
    create_entropy_regularizer,
    create_srip_regularizer,
)
```

## Modules

- `binary_preference.py` — Binary weight preference regularizer + factory
- `tri_state_preference.py` — Ternary weight preference regularizer
- `entropy_regularizer.py` — Shannon entropy targeting + factory
- `soft_orthogonal.py` — Soft orthogonal and orthonormal constraint regularizers
- `srip.py` — SRIP spectral norm regularizer + factory
- `l2_custom.py` — Custom L2 regularizer variant

## Conventions

- All regularizers inherit from `keras.regularizers.Regularizer`
- Must implement `__call__(self, x)` and `get_config()`
- Factory functions (`create_*`) provide convenient construction with defaults
- Fully serializable for model saving/loading

## Testing

Tests in `tests/test_regularizers/`.
