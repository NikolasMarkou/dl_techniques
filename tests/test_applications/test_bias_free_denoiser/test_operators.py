"""Fast CPU unit tests for the measurement operators (ABC + mask family).

Covers success criteria 1-4 (plan.md) for the operators defined in Step 2:
:class:`NullOperator`, :class:`InpaintingOperator`, :class:`RandomPixelsOperator`.
Super-resolution / spectral / compressive-sensing operators (Steps 3-4) are
tested separately.

Checks:

* adjointness ``<measure(x), m> == <x, adjoint(m)>`` (INV-4 orthonormal pair),
* projector idempotency ``project(project(x)) == project(x)`` (INV-4),
* the ``project(ones) == mask`` identity used by :meth:`init_mean`,
* the empty-M degeneration ``project = adjoint = 0`` -> ``d_t = f(y)`` (INV-6),
* output shapes for a batched input,
* the no-dense-matrix guard (INV-3): no operator holds an ``N``-sized array.
"""

import keras
import numpy as np
import pytest

from applications.bias_free_denoiser.operators import (
    IdentityOperator,
    InpaintingOperator,
    MaskOperator,
    MeasurementOperator,
    NullOperator,
    RandomPixelsOperator,
    SpectralDeblurOperator,
    SuperResolutionOperator,
)

# Small, batch>1 shape to catch batch-broadcast bugs while staying CPU-fast.
IMAGE_SHAPE = (32, 32, 3)
BATCH = 2
N = IMAGE_SHAPE[0] * IMAGE_SHAPE[1] * IMAGE_SHAPE[2]  # 3072


def _np(t) -> np.ndarray:
    """Materialize a keras/backend tensor as a NumPy array."""
    return keras.ops.convert_to_numpy(t)


def _rand(shape, seed) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def _inner(a, b) -> float:
    """Full-tensor Frobenius inner product ``<a, b> = sum(a * b)``."""
    return float(np.sum(_np(a) * _np(b)))


def _mask_operators():
    """The two mask operators exercised in this step (fresh instances)."""
    return [
        InpaintingOperator(IMAGE_SHAPE, block_size=(8, 8)),
        RandomPixelsOperator(IMAGE_SHAPE, keep_ratio=0.3, seed=0),
    ]


def _all_operators():
    """Every operator defined in Step 2 (Null + mask family)."""
    return [NullOperator()] + _mask_operators()


class TestAdjointness:
    @pytest.mark.parametrize("op", _all_operators())
    def test_adjointness(self, op):
        # Same-shape measurement domain: draw independent x and m in R^{BxHxWxC}.
        x = _rand((BATCH, *IMAGE_SHAPE), seed=1)
        m = _rand((BATCH, *IMAGE_SHAPE), seed=2)
        lhs = _inner(op.measure(x), m)          # <M^T x, m>
        rhs = _inner(x, op.adjoint(m))          # <x, M m>
        scale = max(1.0, abs(lhs), abs(rhs))
        assert abs(lhs - rhs) <= 1e-4 * scale


class TestProjector:
    @pytest.mark.parametrize("op", _all_operators())
    def test_projector_idempotent(self, op):
        x = _rand((BATCH, *IMAGE_SHAPE), seed=3)
        p1 = _np(op.project(x))
        p2 = _np(op.project(op.project(x)))
        assert np.max(np.abs(p2 - p1)) <= 1e-4

    @pytest.mark.parametrize("op", _mask_operators())
    def test_orthonormal_projector(self, op):
        x = _rand((BATCH, *IMAGE_SHAPE), seed=4)
        # For mask ops, project == measure exactly.
        np.testing.assert_allclose(_np(op.project(x)), _np(op.measure(x)), atol=1e-6)
        # project(ones) recovers the mask (broadcast over batch) — the F2 identity
        # that init_mean relies on.
        ones = np.ones((BATCH, *IMAGE_SHAPE), dtype=np.float32)
        proj_ones = _np(op.project(ones))
        expected = np.broadcast_to(op.mask, (BATCH, *IMAGE_SHAPE))
        np.testing.assert_allclose(proj_ones, expected, atol=1e-6)


class TestNullOperator:
    def test_null_operator_degenerates_to_score(self):
        op = NullOperator()
        x = _rand((BATCH, *IMAGE_SHAPE), seed=5)
        m = _rand((BATCH, *IMAGE_SHAPE), seed=6)
        # project = 0 and adjoint = 0 so d_t = (I - project)f(y) + adjoint(...) = f(y).
        np.testing.assert_allclose(_np(op.project(x)), 0.0, atol=1e-7)
        np.testing.assert_allclose(_np(op.adjoint(m)), 0.0, atol=1e-7)
        np.testing.assert_allclose(_np(op.measure(x)), 0.0, atol=1e-7)
        # Explicit d_t assembly with a stub score f(y): must equal f(y) exactly.
        f_y = _rand((BATCH, *IMAGE_SHAPE), seed=7)
        d_t = (f_y - _np(op.project(f_y))) + _np(op.adjoint(m - _np(op.measure(x))))
        np.testing.assert_allclose(d_t, f_y, atol=1e-6)

    def test_null_init_mean_is_constant_c0_field(self):
        for c0 in (0.0, 0.25):
            op = NullOperator(c0=c0)
            template = _rand((BATCH, *IMAGE_SHAPE), seed=8)
            init = _np(op.init_mean(template))
            assert init.shape == template.shape
            np.testing.assert_allclose(init, c0, atol=1e-7)

    def test_identity_operator_is_null_alias(self):
        assert IdentityOperator is NullOperator
        assert isinstance(IdentityOperator(), MeasurementOperator)


class TestInitMean:
    def test_inpainting_init_mean_exact(self):
        for c0 in (0.0, 0.1):
            op = InpaintingOperator(IMAGE_SHAPE, block_size=(8, 8), c0=c0)
            x_true = _rand((BATCH, *IMAGE_SHAPE), seed=9)
            measurements = _np(op.measure(x_true))  # mask * x_true
            init = _np(op.init_mean(measurements))
            mask = np.broadcast_to(op.mask, (BATCH, *IMAGE_SHAPE))
            expected = mask * x_true + (1.0 - mask) * c0
            np.testing.assert_allclose(init, expected, atol=1e-6)


class TestShapes:
    @pytest.mark.parametrize("op", _all_operators())
    def test_shapes(self, op):
        x = _rand((BATCH, *IMAGE_SHAPE), seed=10)
        m = _rand((BATCH, *IMAGE_SHAPE), seed=11)
        assert _np(op.measure(x)).shape == (BATCH, *IMAGE_SHAPE)
        assert _np(op.adjoint(m)).shape == (BATCH, *IMAGE_SHAPE)
        assert _np(op.project(x)).shape == (BATCH, *IMAGE_SHAPE)
        assert _np(op.init_mean(m)).shape == (BATCH, *IMAGE_SHAPE)


class TestNoDenseMatrix:
    def test_no_dense_matrix(self):
        # INV-3: no operator may hold an array with any dimension == N = H*W*C,
        # and mask arrays must stay [H,W,*] (ndim <= 3), never [N,N] / [n,N].
        for op in _all_operators():
            for name, val in vars(op).items():
                if isinstance(val, np.ndarray):
                    assert N not in val.shape, (
                        f"{type(op).__name__}.{name} has a dim == N ({N}): "
                        f"shape {val.shape} (dense-matrix regression, INV-3)"
                    )
                    assert val.ndim <= 3, (
                        f"{type(op).__name__}.{name} is {val.ndim}-D; masks must "
                        f"be [H,W,*]"
                    )
        # Mask operators specifically expose a small [H,W,*] mask.
        for op in _mask_operators():
            assert isinstance(op, MaskOperator)
            assert op.mask.ndim <= 3
            assert op.mask.shape[:2] == IMAGE_SHAPE[:2]
            assert N not in op.mask.shape


class TestMaskValidation:
    def test_non_binary_mask_rejected(self):
        with pytest.raises(ValueError):
            MaskOperator(np.full((8, 8, 1), 0.5, dtype=np.float32))

    def test_bad_ndim_mask_rejected(self):
        with pytest.raises(ValueError):
            MaskOperator(np.ones((2, 8, 8, 1), dtype=np.float32))

    def test_keep_ratio_range_rejected(self):
        with pytest.raises(ValueError):
            RandomPixelsOperator(IMAGE_SHAPE, keep_ratio=0.0)

    def test_block_too_large_rejected(self):
        with pytest.raises(ValueError):
            InpaintingOperator(IMAGE_SHAPE, block_size=(64, 64))

    def test_random_pixels_reproducible(self):
        a = RandomPixelsOperator(IMAGE_SHAPE, keep_ratio=0.3, seed=42)
        b = RandomPixelsOperator(IMAGE_SHAPE, keep_ratio=0.3, seed=42)
        np.testing.assert_array_equal(a.mask, b.mask)


# =====================================================================
# Step 3: Super-resolution + spectral-deblur operators (HIGH-RISK gate).
#
# The primary gate is ``M^T M ~= I`` (measure(adjoint(m)) == m). For the
# COMPLEX spectral operator this holds only on VALID measurements
# ``m = measure(x)`` (Hermitian-symmetric masked spectra) — the real-signal
# adjoint ``real(idft(...))`` annihilates the anti-Hermitian part of an
# arbitrary complex m. See SpectralDeblurOperator.adjoint DECISION anchor.
# =====================================================================

# Super-res shape: HxW divisible by factor.
SR_SHAPE = (16, 16, 3)
SR_FACTOR = 4
# Spectral shape.
SP_SHAPE = (32, 32, 3)
SP_KEEP = 0.25


def _crand(shape, seed):
    """Random COMPLEX64 array (independent real/imag standard normals)."""
    g = np.random.default_rng(seed)
    return (g.standard_normal(shape) + 1j * g.standard_normal(shape)).astype(np.complex64)


class TestSuperResolutionOperator:
    def test_superres_shapes(self):
        op = SuperResolutionOperator(SR_SHAPE, factor=SR_FACTOR)
        x = _rand((BATCH, *SR_SHAPE), seed=20)
        h, w, c = SR_SHAPE
        lr = (BATCH, h // SR_FACTOR, w // SR_FACTOR, c)
        m = _rand(lr, seed=21)
        assert _np(op.measure(x)).shape == lr
        assert _np(op.adjoint(m)).shape == (BATCH, *SR_SHAPE)
        assert _np(op.project(x)).shape == (BATCH, *SR_SHAPE)
        assert _np(op.init_mean(m)).shape == (BATCH, *SR_SHAPE)

    def test_superres_orthonormal(self):
        # M^T M ~= I on the measurement side: measure(adjoint(m)) == m. This is
        # the STOP-IF gate for the corrected 1/k weighting.
        op = SuperResolutionOperator(SR_SHAPE, factor=SR_FACTOR)
        h, w, c = SR_SHAPE
        m = _rand((BATCH, h // SR_FACTOR, w // SR_FACTOR, c), seed=22)
        recon = _np(op.measure(op.adjoint(m)))
        err = np.max(np.abs(recon - m))
        assert err <= 1e-4, f"super-res M^T M err {err:.3e} > 1e-4"

    def test_superres_adjointness(self):
        op = SuperResolutionOperator(SR_SHAPE, factor=SR_FACTOR)
        h, w, c = SR_SHAPE
        x = _rand((BATCH, *SR_SHAPE), seed=23)
        m = _rand((BATCH, h // SR_FACTOR, w // SR_FACTOR, c), seed=24)
        lhs = _inner(op.measure(x), m)
        rhs = _inner(x, op.adjoint(m))
        scale = max(1.0, abs(lhs), abs(rhs))
        assert abs(lhs - rhs) <= 1e-4 * scale

    def test_superres_projector_idempotent(self):
        op = SuperResolutionOperator(SR_SHAPE, factor=SR_FACTOR)
        x = _rand((BATCH, *SR_SHAPE), seed=25)
        p1 = _np(op.project(x))
        p2 = _np(op.project(op.project(x)))
        assert np.max(np.abs(p2 - p1)) <= 1e-4

    def test_superres_project_equals_meanblock(self):
        op = SuperResolutionOperator(SR_SHAPE, factor=SR_FACTOR)
        x = _rand((BATCH, *SR_SHAPE), seed=26)
        proj = _np(op.project(x))
        # project == adjoint(measure(x))
        adj_meas = _np(op.adjoint(op.measure(x)))
        np.testing.assert_allclose(proj, adj_meas, atol=1e-5)
        # project == explicit block-mean broadcast
        h, w, c = SR_SHAPE
        k = SR_FACTOR
        blocks = x.reshape(BATCH, h // k, k, w // k, k, c)
        mean = blocks.mean(axis=(2, 4), keepdims=True)
        expected = np.tile(mean, (1, 1, k, 1, k, 1)).reshape(BATCH, h, w, c)
        np.testing.assert_allclose(proj, expected, atol=1e-5)

    def test_superres_project_ones(self):
        # Every pixel is in a block whose mean-of-ones is 1 -> project(ones)=1.
        op = SuperResolutionOperator(SR_SHAPE, factor=SR_FACTOR)
        ones = np.ones((BATCH, *SR_SHAPE), dtype=np.float32)
        np.testing.assert_allclose(_np(op.project(ones)), 1.0, atol=1e-6)

    def test_superres_init_mean_is_adjoint(self):
        # project(ones)=1 -> init_mean reduces to adjoint(measurements).
        op = SuperResolutionOperator(SR_SHAPE, factor=SR_FACTOR, c0=0.3)
        h, w, c = SR_SHAPE
        m = _rand((BATCH, h // SR_FACTOR, w // SR_FACTOR, c), seed=27)
        np.testing.assert_allclose(
            _np(op.init_mean(m)), _np(op.adjoint(m)), atol=1e-5
        )

    def test_superres_bad_factor_rejected(self):
        with pytest.raises(ValueError):
            SuperResolutionOperator((16, 16, 3), factor=5)  # 16 % 5 != 0


class TestSpectralDeblurOperator:
    def test_spectral_unitary_sanity(self):
        # idft(dft(x)) ~= x and Parseval ||dft(x)|| ~= ||x|| for random real x.
        op = SpectralDeblurOperator(SP_SHAPE, keep_fraction=SP_KEEP)
        x = _rand((BATCH, *SP_SHAPE), seed=30)
        roundtrip = _np(op.idft(op.dft(x)))
        np.testing.assert_allclose(roundtrip.real, x, atol=1e-4)
        assert np.max(np.abs(roundtrip.imag)) <= 1e-4
        spec = _np(op.dft(x))
        assert abs(np.linalg.norm(spec) - np.linalg.norm(x)) <= 1e-3 * np.linalg.norm(x)

    def test_spectral_shapes(self):
        op = SpectralDeblurOperator(SP_SHAPE, keep_fraction=SP_KEEP)
        x = _rand((BATCH, *SP_SHAPE), seed=31)
        meas = op.measure(x)
        assert _np(meas).shape == (BATCH, *SP_SHAPE)
        assert _np(meas).dtype == np.complex64
        assert _np(op.adjoint(meas)).shape == (BATCH, *SP_SHAPE)
        assert _np(op.project(x)).shape == (BATCH, *SP_SHAPE)
        assert _np(op.init_mean(meas)).shape == (BATCH, *SP_SHAPE)

    def test_spectral_orthonormal(self):
        # STOP-IF gate: M^T M ~= I on the RANGE of M (valid measurements).
        # m = measure(x) is a Hermitian-symmetric masked spectrum; the real-
        # signal adjoint recovers it exactly. (Arbitrary random complex m would
        # NOT satisfy this because real(idft(.)) kills the anti-Hermitian part.)
        op = SpectralDeblurOperator(SP_SHAPE, keep_fraction=SP_KEEP)
        x = _rand((BATCH, *SP_SHAPE), seed=32)
        m = op.measure(x)
        recon = _np(op.measure(op.adjoint(m)))
        err = np.max(np.abs(recon - _np(m)))
        assert err <= 1e-4, f"spectral M^T M err {err:.3e} > 1e-4"

    def test_spectral_adjointness(self):
        # Real-part Hermitian inner product: real(<measure(x), m>) == <x, adjoint(m)>
        # holds for ANY complex m (real x). Derived from M being real-linear into
        # a real signal: <Lambda F x, m> real part == <x, real(F^H Lambda m)>.
        op = SpectralDeblurOperator(SP_SHAPE, keep_fraction=SP_KEEP)
        x = _rand((BATCH, *SP_SHAPE), seed=33)
        m = _crand((BATCH, *SP_SHAPE), seed=34)
        meas = _np(op.measure(x))
        lhs = float(np.real(np.sum(np.conj(meas) * m)))
        rhs = float(np.sum(x * _np(op.adjoint(m))))
        scale = max(1.0, abs(lhs), abs(rhs))
        assert abs(lhs - rhs) <= 1e-4 * scale

    def test_spectral_projector_idempotent(self):
        op = SpectralDeblurOperator(SP_SHAPE, keep_fraction=SP_KEEP)
        x = _rand((BATCH, *SP_SHAPE), seed=35)
        p1 = _np(op.project(x))
        p2 = _np(op.project(op.project(x)))
        assert np.max(np.abs(p2 - p1)) <= 1e-4

    def test_spectral_project_real(self):
        # INV-5: project output is strictly real (float32, no imag part).
        op = SpectralDeblurOperator(SP_SHAPE, keep_fraction=SP_KEEP)
        x = _rand((BATCH, *SP_SHAPE), seed=36)
        proj = _np(op.project(x))
        assert not np.iscomplexobj(proj)

    def test_spectral_project_ones(self):
        # DC-containing low-pass reconstructs a constant field exactly.
        op = SpectralDeblurOperator(SP_SHAPE, keep_fraction=SP_KEEP)
        ones = np.ones((BATCH, *SP_SHAPE), dtype=np.float32)
        np.testing.assert_allclose(_np(op.project(ones)), 1.0, atol=1e-4)

    def test_spectral_bad_keep_fraction_rejected(self):
        with pytest.raises(ValueError):
            SpectralDeblurOperator(SP_SHAPE, keep_fraction=0.0)
