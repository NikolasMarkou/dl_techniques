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
    PolarInitializer,
    GaborFiltersInitializer,
    create_gabor_depthwise_conv2d,
)
```

## Modules

- `orthonormal_initializer.py` — `OrthonormalInitializer`: orthonormal row matrices for signal norm preservation
- `he_orthonormal_initializer.py` — `HeOrthonormalInitializer`: He variance scaling + orthonormalization for ReLU networks
- `hypersphere_orthogonal_initializer.py` — `OrthogonalHypersphereInitializer`: orthogonal vectors on a hypersphere
- `haar_wavelet_initializer.py` — `HaarWaveletInitializer`: fixed 2D Haar wavelet filters for conv layers; also provides `create_haar_depthwise_conv2d` factory
- `polar_initializer.py` — `PolarInitializer`: exact per-vector L2 norm with a uniform-on-sphere direction ("equinorm" init); polar-coordinate sampling (PolarQuant Lemma 2)
- `gabor_filters_initializer.py` — `GaborFiltersInitializer`: deterministic Gabor filter-bank initialization (Ozbulak & Ekenel); also provides `create_gabor_depthwise_conv2d` factory (per-channel/depthwise, no cross-channel mixing; output = `in_channels * filters`; follow with a 1x1 Conv2D for a specific output count)

## Conventions

- All initializers inherit from `keras.initializers.Initializer`
- Must implement `__call__(self, shape, dtype=None)` and `get_config()` for serialization
- Mathematically principled — each is based on a specific theoretical property

## Testing

Tests in `tests/test_initializers/`.
