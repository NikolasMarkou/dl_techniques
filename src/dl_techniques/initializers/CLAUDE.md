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
    create_gabor_conv2d,
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
- `hypersphere_orthogonal_initializer.py` — `OrthogonalHypersphereInitializer`: vectors on a hypersphere of radius `radius`, mutually orthogonal where possible. The flattened matrix is `(prod(shape[:-1]), shape[-1])` and is NOT transposed, so a Dense kernel `(input_dim, units)` is `input_dim` vectors in `units` dimensions — every narrowing projection and effectively every Conv2D exceeds `latent_dim`. That regime stacks `ceil(nv/d)` independent orthonormal bases (`cond` 1.0-1.41, `1/ceil(nv/d)` of pairs exactly orthogonal) instead of the uniform sampling it used to degrade to (`cond` 2.92, 0.2%); `fallback='uniform'` restores the old path. The QR is sign-corrected by `sign(diag(R))` like `he_orthonormal_initializer.py` — without it `Q[0,0]` was negative in 2000 of 2000 draws. It is the DEFAULT initializer for `OrthoBlock`, `OrthoGLUFFN` and `NeuroGrid`, so this regime is the ordinary case, not an edge case
- `haar_wavelet_initializer.py` — `HaarWaveletInitializer`: the fixed, ORTHONORMAL 2D Haar filter bank for conv layers (every tap +/- 0.5, Gram matrix = identity, so energy-preserving and inverted by its own transpose at `scale=1.0`); output slot `j` holds sub-band `j % 4` (`LL, LH, HL, HH`) for EVERY input channel, so a `DepthwiseConv2D` output channel `i*cm + j` is sub-band `j` of input `i`. Also provides the `create_haar_depthwise_conv2d` factory, which rejects odd spatial dimensions (a stride-2 `valid` decomposition would drop the last row/column)
- `polar_initializer.py` — `PolarInitializer`: exact per-vector L2 norm with a uniform-on-sphere direction ("equinorm" init); polar-coordinate sampling (PolarQuant Lemma 2). `axis=None` (default) reduces over EVERY AXIS BUT THE LAST — the fan-in block — so the He-equivalent `sqrt(2)` target is correct for a Conv2D kernel as well as a Dense one; a scalar axis on a `(3,3,64,128)` kernel gave each output unit a fan-in energy of 384 instead of 2. It does NOT give dynamical isometry (the spectrum stays Marchenko-Pastur; use `keras.initializers.Orthogonal`), and the benefit over He shrinks as `1/sqrt(2*fan_in)` (17.9% norm spread at fan_in=16, 1.1% at 4096)
- `gabor_filters_initializer.py` — `GaborFiltersInitializer`: deterministic Gabor filter-bank initialization (Ozbulak & Ekenel) over a factorized orientation x scale x phase sweep (`sweep="diagonal"` recovers the paper's joint-`linspace` construction), DC-removed and energy-normalized to `sqrt(2/fan_in)` by default. Two factories: `create_gabor_conv2d` (cross-channel, `trainable=True` — the paper's warm start) and `create_gabor_depthwise_conv2d` (per-channel/depthwise, no cross-channel mixing, `trainable=False`; output = `in_channels * filters_per_channel`; follow with a 1x1 Conv2D for a specific output count)
- `linear_up_initializer.py` — `LinearUpInitializer`: THERA heat-field frequency init; 2D frequency vectors uniform over a disk of radius `pi*scale`, producing a `(2, N)` x/y-row matrix
- `identity_plus_noise.py` — `IdentityPlusNoise`: `eye(H) + RandomNormal(stddev, seed)` for SQUARE 2-D shapes only (raises `ValueError` otherwise); `stddev=0` gives the exact identity. A near-identity start for a mixing/coupling matrix meant to begin as a no-op; used by `WaveFieldAttention.field_coupling`
- `clone.py` — `clone_initializer`: returns an INDEPENDENT initializer. A single *seedless* Keras 3 initializer INSTANCE self-assigns a seed and replays it, so reusing one instance across several weights emits the SAME tensor at every matching shape (measured: two `Dense(4)` sharing one instance are bit-identical; two built from the STRING `"glorot_uniform"` are not). Clone per site only where the two weights play DIFFERENT architectural roles. A *seeded* initializer still yields identical clones, deliberately
- `kan_initializer.py` — `KANInitializer`: variance-controlled init for KAN residual (`base_scaler`) and spline (`spline_weight`) roles, implementing the three schemes of Rigas et al., *Initialization Schemes for Kolmogorov-Arnold Networks: An Empirical Study* (arXiv:2509.03417). The `mu_R_*` / `mu_B_*` expectation constants are computed EXACTLY (Gauss-Legendre over `grid_range`, and composite Gauss-Legendre over the host's own Cox-de Boor knot vector) — they were Monte-Carlo estimates whose value moved 7.9% with the RNG seed, plus the proxies `1/(G+1)` and `1.0` which ran 2.3-2.8x high and were blind to `grid_size`. **None of the three schemes is unit-gain**: `power_law` is an empirical grid-search rule that scales as `sqrt(n_in/N)` (gain 0.134 at width 16, 2.14 at 4096), `glorot_inspired` is the only width-independent one (0.264 flat) and the only one derived from a variance argument, and `baseline`'s fixed spline noise makes its gain grow linearly with width (0.171 -> 19.7). Use `expected_forward_gain(n_in, n_out)` rather than assuming depth stability. `N` is pinned to the host's `grid_size + spline_order`, NOT the paper's `G+k+1` (D-001), and the spline branch now REJECTS a last dim that disagrees. Also provides `create_kan_initializers`, which forwards every scheme parameter (it could previously select `baseline` without being able to set `baseline_noise`) and gives the two roles independent random streams (they previously correlated at exactly 1.0000)

## Conventions

- All initializers inherit from `keras.initializers.Initializer`
- Must implement `__call__(self, shape, dtype=None)` and `get_config()` for serialization
- Mathematically principled — each is based on a specific theoretical property

## Testing

Tests in `tests/test_initializers/`.
