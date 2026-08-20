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
    LinearUpInitializer,
    IdentityPlusNoise,
    KANInitializer,
    create_kan_initializers,
    clone_initializer,
)
```

## Modules

- `orthonormal_initializer.py` — `OrthonormalInitializer`: orthonormal row matrices for signal norm preservation
- `he_orthonormal_initializer.py` — `HeOrthonormalInitializer`: He variance scaling + orthonormalization for ReLU networks
- `hypersphere_orthogonal_initializer.py` — `OrthogonalHypersphereInitializer`: orthogonal vectors on a hypersphere
- `haar_wavelet_initializer.py` — `HaarWaveletInitializer`: fixed 2D Haar wavelet filters for conv layers; also provides `create_haar_depthwise_conv2d` factory
- `polar_initializer.py` — `PolarInitializer`: exact per-vector L2 norm with a uniform-on-sphere direction ("equinorm" init); polar-coordinate sampling (PolarQuant Lemma 2)
- `gabor_filters_initializer.py` — `GaborFiltersInitializer`: deterministic Gabor filter-bank initialization (Ozbulak & Ekenel); also provides `create_gabor_depthwise_conv2d` factory (per-channel/depthwise, no cross-channel mixing; output = `in_channels * filters`; follow with a 1x1 Conv2D for a specific output count)
- `linear_up_initializer.py` — `LinearUpInitializer`: THERA heat-field frequency init; 2D frequency vectors uniform over a disk of radius `pi*scale`, producing a `(2, N)` x/y-row matrix
- `identity_plus_noise.py` — `IdentityPlusNoise`: `eye(H) + RandomNormal(stddev, seed)` for SQUARE 2-D shapes only (raises `ValueError` otherwise); `stddev=0` gives the exact identity. A near-identity start for a mixing/coupling matrix meant to begin as a no-op; used by `WaveFieldAttention.field_coupling`
- `clone.py` — `clone_initializer`: returns an INDEPENDENT initializer. A single *seedless* Keras 3 initializer INSTANCE self-assigns a seed and replays it, so reusing one instance across several weights emits the SAME tensor at every matching shape (measured: two `Dense(4)` sharing one instance are bit-identical; two built from the STRING `"glorot_uniform"` are not). Clone per site only where the two weights play DIFFERENT architectural roles — see `plan-2026-08-19T163559-499b6f0e/D-057`. A *seeded* initializer still yields identical clones, deliberately
- `kan_initializer.py` — `KANInitializer`: variance-controlled init for KAN residual (`base_scaler`) and spline (`spline_weight`) roles; also provides `create_kan_initializers`

## Conventions

- All initializers inherit from `keras.initializers.Initializer`
- Must implement `__call__(self, shape, dtype=None)` and `get_config()` for serialization
- Mathematically principled — each is based on a specific theoretical property

## Testing

Tests in `tests/test_initializers/`.
