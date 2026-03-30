# Initializers Package

Advanced Keras weight initializers based on orthogonality, signal processing, and statistical learning theory.

## Public API

```python
from dl_techniques.initializers import (
    OrthonormalInitializer,
    HeOrthonormalInitializer,
    OrthogonalHypersphereInitializer,
    HaarWaveletInitializer,
    create_haar_depthwise_conv2d,
)
```

## Modules

- `orthonormal_initializer.py` — `OrthonormalInitializer`: orthonormal row matrices for signal norm preservation
- `he_orthonormal_initializer.py` — `HeOrthonormalInitializer`: He variance scaling + orthonormalization for ReLU networks
- `hypersphere_orthogonal_initializer.py` — `OrthogonalHypersphereInitializer`: orthogonal vectors on a hypersphere
- `haar_wavelet_initializer.py` — `HaarWaveletInitializer`: fixed 2D Haar wavelet filters for conv layers; also provides `create_haar_depthwise_conv2d` factory

## Conventions

- All initializers inherit from `keras.initializers.Initializer`
- Must implement `__call__(self, shape, dtype=None)` and `get_config()` for serialization
- Mathematically principled — each is based on a specific theoretical property

## Testing

Tests in `tests/test_initializers/`.
