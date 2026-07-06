"""Matched measurement operators for the bias-free denoiser inverse-problem solver.

Each of Kadkhodaie & Simoncelli 2020's problems (prior sampling, inpainting,
random-missing-pixels, super-resolution, spectral deblurring, compressive
sensing) is selected by swapping ONE :class:`MeasurementOperator` subclass into
the shared Algorithm-2 loop. The solver contains no per-problem branching — the
operator is the single load-bearing abstraction (D-001).

This module holds the ABC contract plus the trivial operators built on 0/1
masks (:class:`NullOperator`, :class:`MaskOperator`, :class:`InpaintingOperator`,
:class:`RandomPixelsOperator`). The heavier spectral / super-resolution /
compressive-sensing operators live alongside them in later steps.

Design invariants (plan.md):

* INV-3 (no dense matrices): operators NEVER materialize an array with any
  dimension equal to ``N = H*W*C``. They act on same-shape ``[B, H, W, C]``
  tensors and hold at most an ``[H, W, *]`` mask. This is the whole reason the
  app scales past toy sizes — the old ``samplers.py`` built ``[n_meas, N]`` /
  ``[N, N]`` matrices (~154 GB at 256x256x3).
* INV-4 (orthonormal M): every operator satisfies ``M^T M = I`` so
  ``project = MM^T`` is a genuine idempotent orthogonal projector onto the
  measured subspace. For a 0/1 mask this is automatic (disjoint-support,
  unit-norm columns).
* INV-6 (unified d_t): the solver's update is
  ``d_t = (I - project)(f(y)) + adjoint(measurements - measure(y))``.
  For :class:`NullOperator` (empty M) ``project`` and ``adjoint`` return zeros,
  so ``d_t`` degenerates exactly to ``f(y) = D(y) - y`` — Algorithm-1 prior
  sampling.

Domain: pixels live in ``[-0.5, +0.5]`` with domain center ``c0 = 0.0`` (D-002);
``c0`` is the constant field to which unmeasured signal components are
initialized in :meth:`MeasurementOperator.init_mean`.
"""

import abc
import math
from typing import Optional, Sequence, Tuple, Union

import keras
import numpy as np
import tensorflow as tf

from dl_techniques.utils.logger import logger

# Image tensors are signal-domain ``[B, H, W, C]``; ``keras.ops`` accepts either
# a NumPy array (host-side test input) or a backend tensor transparently.
TensorLike = Union[np.ndarray, "keras.KerasTensor"]

# ``(H, W, C)`` signal shape (batch excluded — operators are batch-agnostic).
ImageShape = Tuple[int, int, int]


def _dct_available() -> bool:
    """Probe whether ``tf.signal.dct``/``idct`` (type-2, ``norm='ortho'``) work.

    The compressive-sensing operator PREFERS a real orthonormal DCT transform,
    but ``tf.signal.dct`` is not guaranteed present/correct across TF builds
    (plan A6/Q3). This runs a tiny round-trip once at import so the operator can
    fall back to the confirmed-available unitary complex-FFT path without ever
    failing the suite for a missing DCT.

    Returns:
        ``True`` iff DCT-II with ``norm='ortho'`` and its inverse exist and
        round-trip on a probe tensor to float precision.
    """
    try:
        probe = tf.constant([[1.0, 2.0, 3.0, 4.0]], dtype=tf.float32)
        fwd = tf.signal.dct(probe, type=2, norm="ortho")
        rt = tf.signal.idct(fwd, type=2, norm="ortho")
        return bool(tf.reduce_max(tf.abs(rt - probe)).numpy() < 1e-4)
    except Exception as exc:  # pragma: no cover - defensive: absent/broken DCT
        logger.warning("tf.signal.dct unavailable (%s); CS uses FFT fallback", exc)
        return False


# Probed ONCE at import: which orthonormal transform the CS operator uses.
_DCT_AVAILABLE: bool = _dct_available()


class MeasurementOperator(abc.ABC):
    """Abstract linear measurement operator ``M`` with an orthonormal column set.

    A measurement operator exposes four methods used by the unified solver:

    * :meth:`measure` applies ``M^T`` (signal -> measurement domain).
    * :meth:`adjoint` applies ``M`` (measurement -> signal domain).
    * :meth:`project` applies ``M M^T`` (signal -> signal orthogonal projector
      onto the measured subspace); idempotent because ``M^T M = I`` (INV-4).
    * :meth:`init_mean` builds the Algorithm-2 initialization mean from a set of
      measurements.

    Subclasses MUST implement :meth:`measure` and :meth:`adjoint`. The base class
    supplies a correct default :meth:`project` (``adjoint(measure(x))``, valid for
    any orthonormal-column ``M``) and a default :meth:`init_mean` using the
    domain-generalized formula below; a subclass overrides these only when a
    cheaper closed form exists.

    Attributes:
        c0: Domain-center constant (``0.0`` for the ``[-0.5, +0.5]`` model, D-002).
            Unmeasured signal components are initialized to this value.
    """

    def __init__(self, c0: float = 0.0) -> None:
        """Store the domain-center constant.

        Args:
            c0: Domain center of the pixel space (``0.0`` for this model, D-002).
        """
        self.c0: float = float(c0)

    # ------------------------------------------------------------------
    # Contract — subclasses implement the two adjoint half-maps.
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def measure(self, x: TensorLike) -> "keras.KerasTensor":
        """Apply ``M^T``: map a signal-domain tensor to the measurement domain.

        Args:
            x: A signal-domain tensor ``[B, H, W, C]``.

        Returns:
            The measurement-domain tensor ``m``.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def adjoint(self, m: TensorLike) -> "keras.KerasTensor":
        """Apply ``M``: map a measurement-domain tensor back to signal domain.

        Args:
            m: A measurement-domain tensor.

        Returns:
            A signal-domain tensor ``[B, H, W, C]``.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Defaults — correct for any orthonormal-column M; override for speed.
    # ------------------------------------------------------------------

    def project(self, x: TensorLike) -> "keras.KerasTensor":
        """Apply ``M M^T``, the orthogonal projector onto the measured subspace.

        Default implementation composes ``adjoint(measure(x))``. This is exactly
        ``M M^T`` and is idempotent whenever ``M^T M = I`` (INV-4). Subclasses
        with a cheaper closed form (e.g. a diagonal mask) override this.

        Args:
            x: A signal-domain tensor ``[B, H, W, C]``.

        Returns:
            The projected signal-domain tensor, same shape as ``x``.
        """
        return self.adjoint(self.measure(x))

    def init_mean(self, measurements: TensorLike) -> "keras.KerasTensor":
        """Build the Algorithm-2 initialization mean from measurements.

        # DECISION plan_2026-07-06_d6b88914/D-002: the paper's init uses a literal
        # `0.5 * (I - MM^T) e` DC term (for `[0,1]` data). THIS checkpoint trains
        # in `[-0.5, +0.5]` with domain center `c0 = 0.0`, so the constant is
        # `c0`, NOT 0.5 — do not hardcode 0.5 here or every unmeasured pixel gets
        # DC-biased. The generalized formula keeps `(1 - project(ones))` rather
        # than assuming `(1 - mask)`, because for spectral/CS operators
        # `project(ones)` is not a plain 0/1 field. See decisions.md D-002.

        Computes ``init_mean = c0 * (1 - project(ones)) + adjoint(measurements)``:
        the measured component is placed at ``adjoint(measurements)`` and every
        unmeasured component is set to the domain center ``c0``.

        Args:
            measurements: A measurement-domain tensor (its ``adjoint`` supplies the
                target signal shape).

        Returns:
            A signal-domain tensor ``[B, H, W, C]`` — the init distribution mean
            (the solver adds ``sigma_0`` Gaussian noise on top).
        """
        signal = self.adjoint(measurements)
        ones = keras.ops.ones_like(signal)
        unmeasured = keras.ops.subtract(1.0, self.project(ones))
        return keras.ops.add(keras.ops.multiply(self.c0, unmeasured), signal)


class NullOperator(MeasurementOperator):
    """Empty measurement operator for unconstrained prior sampling (F2 §0).

    "Null" (not "identity"): the measurement set is EMPTY, so there is nothing to
    measure and no measured subspace. Every half-map returns zeros.

    # DECISION plan_2026-07-06_d6b88914/D-001: do NOT implement this as identity
    # passthroughs. The unified update `d_t = (I - project)(f(y)) + adjoint(
    # measurements - measure(y))` must degenerate to `d_t = f(y)` (Algorithm-1)
    # for the empty-M case (INV-6). That requires `project = 0` AND `adjoint = 0`
    # (giving `d_t = (I - 0)f(y) + 0 = f(y)`), NOT `project = I` / `adjoint = I`
    # (which would over-constrain to the input). `init_mean` collapses to the
    # constant `c0` field. Verified end-to-end by the Step-5 solver's
    # Identity=Algorithm-1 convergence test. See decisions.md D-001.

    Note:
        :class:`IdentityOperator` is a public alias of this class; both refer to
        the empty-measurement / identity-prior case used for prior sampling.
    """

    def measure(self, x: TensorLike) -> "keras.KerasTensor":
        """Return zeros (no measurement is taken).

        Args:
            x: A signal-domain tensor ``[B, H, W, C]``.

        Returns:
            A zero tensor with the same shape as ``x``.
        """
        return keras.ops.zeros_like(x)

    def adjoint(self, m: TensorLike) -> "keras.KerasTensor":
        """Return zeros: there is no measured component to scatter back.

        Args:
            m: A measurement-domain tensor.

        Returns:
            A zero tensor with the same shape as ``m``.
        """
        return keras.ops.zeros_like(m)

    def project(self, x: TensorLike) -> "keras.KerasTensor":
        """Return zeros: the measured subspace is empty (the zero projector).

        Args:
            x: A signal-domain tensor ``[B, H, W, C]``.

        Returns:
            A zero tensor with the same shape as ``x``.
        """
        return keras.ops.zeros_like(x)

    def init_mean(self, measurements: TensorLike) -> "keras.KerasTensor":
        """Return the constant ``c0`` field shaped like ``measurements``.

        With ``project = 0`` and ``adjoint = 0`` the base formula reduces to
        ``c0 * ones``; computed directly here so prior sampling starts from a flat
        domain-center field plus (solver-added) noise.

        Args:
            measurements: A signal-shaped template tensor (its shape is used).

        Returns:
            A tensor of ``c0`` with the same shape as ``measurements``.
        """
        return keras.ops.multiply(self.c0, keras.ops.ones_like(measurements))


# Public alias: prior sampling reads more naturally as an "identity prior".
IdentityOperator = NullOperator


class MaskOperator(MeasurementOperator):
    """Diagonal 0/1 masking operator (base for inpainting / random pixels, F2 §1).

    A binary mask ``[H, W, 1]`` or ``[H, W, C]`` (``1`` = kept/measured) defines a
    diagonal ``M`` whose columns are disjoint-support one-hot vectors — trivially
    orthonormal, so ``M^T M = I`` and ``project = MM^T`` is the mask itself. The
    measurement is represented in the SAME shape as the signal (measured pixels
    kept, others zeroed) rather than a compacted vector, which keeps every op
    reshape-free and batch-friendly (F2 §1).

    Attributes:
        mask: A float32 ``[H, W, 1]`` or ``[H, W, C]`` array of 0/1, broadcast
            over the batch. Built ONCE at construction (INV-3: no per-call
            allocation, no branching on tensor values).
    """

    def __init__(self, mask: np.ndarray, c0: float = 0.0) -> None:
        """Wrap a precomputed 0/1 mask.

        Args:
            mask: A 0/1 array of shape ``[H, W]``, ``[H, W, 1]`` or ``[H, W, C]``.
                A 2-D ``[H, W]`` mask is promoted to ``[H, W, 1]`` (broadcast over
                channels).
            c0: Domain center for unmeasured pixels (default ``0.0``, D-002).

        Raises:
            ValueError: If ``mask`` is not 2-D or 3-D, or holds values other than
                ``0`` / ``1``.
        """
        super().__init__(c0=c0)
        m = np.asarray(mask, dtype=np.float32)
        if m.ndim == 2:
            m = m[:, :, None]
        if m.ndim != 3:
            raise ValueError(
                f"mask must be [H,W], [H,W,1] or [H,W,C]; got shape {m.shape}"
            )
        unique = np.unique(m)
        if not np.all(np.isin(unique, (0.0, 1.0))):
            raise ValueError(f"mask must be binary 0/1; got values {unique[:8]}")
        self.mask: np.ndarray = m
        kept = float(m.mean())
        logger.info(
            "MaskOperator: mask %s, %.1f%% pixels kept (c0=%.3f)",
            tuple(m.shape), 100.0 * kept, self.c0,
        )

    def measure(self, x: TensorLike) -> "keras.KerasTensor":
        """Apply ``M^T`` = keep measured pixels, zero the rest: ``mask * x``.

        Args:
            x: A signal-domain tensor ``[B, H, W, C]``.

        Returns:
            The masked tensor ``mask * x``, same shape as ``x``.
        """
        return keras.ops.multiply(self.mask, x)

    def adjoint(self, m: TensorLike) -> "keras.KerasTensor":
        """Apply ``M`` = re-mask (idempotent scatter): ``mask * m``.

        The same-shape masked representation makes ``M`` a re-application of the
        mask, keeping the result in the measured subspace regardless of what was
        stored outside it.

        Args:
            m: A same-shape measurement-domain tensor ``[B, H, W, C]``.

        Returns:
            ``mask * m``, same shape as ``m``.
        """
        return keras.ops.multiply(self.mask, m)

    def project(self, x: TensorLike) -> "keras.KerasTensor":
        """Apply ``M M^T`` = the diagonal 0/1 projector: ``mask * x`` (== measure).

        Args:
            x: A signal-domain tensor ``[B, H, W, C]``.

        Returns:
            ``mask * x``, same shape as ``x``.
        """
        return keras.ops.multiply(self.mask, x)

    def init_mean(self, measurements: TensorLike) -> "keras.KerasTensor":
        """Init mean ``mask * measurements + (1 - mask) * c0`` (F2 §1).

        Measured pixels start at their observed value; unmeasured pixels start at
        the domain center ``c0``. (Equivalent to the base-class formula for a 0/1
        mask, written directly for clarity.)

        Args:
            measurements: The same-shape masked observation ``[B, H, W, C]``.

        Returns:
            The init-distribution mean, same shape as ``measurements``.
        """
        measured = keras.ops.multiply(self.mask, measurements)
        unmeasured = keras.ops.multiply(1.0 - self.mask, self.c0)
        return keras.ops.add(measured, unmeasured)


class InpaintingOperator(MaskOperator):
    """Block inpainting: a centered rectangular region is missing (F2 §1).

    Builds a mask that is ``0`` inside a centered ``block_size`` rectangle
    (missing) and ``1`` everywhere else (observed). The block is dropped for the
    whole pixel (mask shape ``[H, W, 1]``, broadcast over channels).
    """

    def __init__(
        self,
        image_shape: ImageShape,
        block_size: Union[int, Sequence[int]],
        c0: float = 0.0,
    ) -> None:
        """Build a centered-block missing mask.

        Args:
            image_shape: Signal shape ``(H, W, C)``.
            block_size: Missing-region size, an int (square) or ``(bh, bw)``.
            c0: Domain center for missing pixels (default ``0.0``, D-002).

        Raises:
            ValueError: If the block does not fit inside the image.
        """
        h, w, _ = image_shape
        if isinstance(block_size, int):
            bh, bw = block_size, block_size
        else:
            bh, bw = int(block_size[0]), int(block_size[1])
        if bh > h or bw > w or bh < 0 or bw < 0:
            raise ValueError(
                f"block_size {(bh, bw)} does not fit in image {(h, w)}"
            )
        mask = np.ones((h, w, 1), dtype=np.float32)
        top, left = (h - bh) // 2, (w - bw) // 2
        mask[top: top + bh, left: left + bw, :] = 0.0
        super().__init__(mask=mask, c0=c0)


class RandomPixelsOperator(MaskOperator):
    """Random missing pixels: keep a ``keep_ratio`` fraction at random (F2 §1).

    Builds a random 0/1 mask keeping (in expectation) ``keep_ratio`` of the
    pixels. The SAME mask is used across channels by default (a whole pixel is
    kept or dropped), matching the paper's per-pixel measurement model.
    """

    def __init__(
        self,
        image_shape: ImageShape,
        keep_ratio: float = 0.5,
        *,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        c0: float = 0.0,
    ) -> None:
        """Build a random per-pixel keep mask.

        Args:
            image_shape: Signal shape ``(H, W, C)``.
            keep_ratio: Fraction of pixels to keep (in ``(0, 1]``).
            seed: Optional seed for a fresh ``numpy`` generator (reproducibility).
            rng: Optional pre-seeded ``numpy.random.Generator`` (takes precedence
                over ``seed``).
            c0: Domain center for dropped pixels (default ``0.0``, D-002).

        Raises:
            ValueError: If ``keep_ratio`` is not in ``(0, 1]``.
        """
        if not (0.0 < keep_ratio <= 1.0):
            raise ValueError(f"keep_ratio must be in (0, 1]; got {keep_ratio}")
        h, w, _ = image_shape
        generator = rng if rng is not None else np.random.default_rng(seed)
        # One draw per PIXEL (broadcast over channels), thresholded to 0/1.
        keep = (generator.random((h, w, 1)) < keep_ratio).astype(np.float32)
        super().__init__(mask=keep, c0=c0)


class SuperResolutionOperator(MeasurementOperator):
    """kxk block-averaging downsample with an ORTHONORMAL-column ``M`` (F2 §2).

    Super-resolution measures a low-resolution image by averaging each
    non-overlapping ``k x k`` block. The naive box-average (weight ``1/k^2`` per
    element) has row norm ``1/k != 1``, so ``M`` would NOT have orthonormal
    columns and ``project = MM^T`` would not be idempotent (INV-4). The corrected
    orthonormal pair (F2 §2, verified) uses weight ``1/k`` per element:

    * :meth:`measure` = ``k * mean_block(x) = (1/k) * sum_block(x)`` — row norm
      ``sqrt(k^2 * (1/k)^2) = 1``.
    * :meth:`adjoint` = broadcast ``v / k`` to each of the ``k^2`` block positions.
    * :meth:`project` = ``adjoint(measure(x))`` = replace each block by its mean
      (the ``k`` and ``1/k`` factors cancel — NO extra scaling), implemented
      directly as reshape -> mean -> broadcast for numerical cleanliness.

    With this pair ``measure(adjoint(m)) = m`` exactly (``M^T M = I``) and
    ``project`` is idempotent. ``project(ones) = 1`` everywhere (every pixel lies
    in a block whose mean-of-ones is 1), so :meth:`init_mean` reduces to
    ``adjoint(measurements)``.

    All ops are pure reshape / mean / tile — O(N), no dense matrix (INV-3).

    Attributes:
        h, w, c: Signal spatial / channel dims.
        k: Downsample factor (block edge length).
    """

    def __init__(self, image_shape: ImageShape, factor: int = 4, c0: float = 0.0) -> None:
        """Build a super-resolution (block-averaging) operator.

        Args:
            image_shape: Signal shape ``(H, W, C)``.
            factor: Block edge length ``k`` (downsample factor). ``H`` and ``W``
                MUST be divisible by ``k``.
            c0: Domain center for unmeasured components (default ``0.0``, D-002).

        Raises:
            ValueError: If ``factor < 1`` or ``H``/``W`` are not divisible by it.
        """
        super().__init__(c0=c0)
        h, w, c = int(image_shape[0]), int(image_shape[1]), int(image_shape[2])
        k = int(factor)
        if k < 1:
            raise ValueError(f"factor must be >= 1; got {k}")
        if h % k != 0 or w % k != 0:
            raise ValueError(
                f"image {(h, w)} not divisible by factor {k} "
                f"(super-res needs non-overlapping k x k blocks)"
            )
        self.h, self.w, self.c, self.k = h, w, c, k
        logger.info(
            "SuperResolutionOperator: %s, factor %d -> LR %s",
            (h, w, c), k, (h // k, w // k, c),
        )

    def measure(self, x: TensorLike) -> "keras.KerasTensor":
        """Apply ``M^T`` = ``k * mean_block(x)`` -> LR image ``[B, H/k, W/k, C]``.

        Args:
            x: A signal-domain tensor ``[B, H, W, C]``.

        Returns:
            The low-resolution measurement ``[B, H/k, W/k, C]``.
        """
        k, h, w, c = self.k, self.h, self.w, self.c
        blocks = keras.ops.reshape(x, (-1, h // k, k, w // k, k, c))
        return keras.ops.multiply(float(k), keras.ops.mean(blocks, axis=(2, 4)))

    def adjoint(self, m: TensorLike) -> "keras.KerasTensor":
        """Apply ``M`` = broadcast ``m / k`` into every block position (F2 §2).

        Args:
            m: A low-resolution measurement ``[B, H/k, W/k, C]``.

        Returns:
            A signal-domain tensor ``[B, H, W, C]``; each pixel receives ``m/k``.
        """
        k, h, w, c = self.k, self.h, self.w, self.c
        v = keras.ops.divide(m, float(k))
        v = keras.ops.reshape(v, (-1, h // k, 1, w // k, 1, c))
        v = keras.ops.tile(v, (1, 1, k, 1, k, 1))
        return keras.ops.reshape(v, (-1, h, w, c))

    def project(self, x: TensorLike) -> "keras.KerasTensor":
        """Apply ``M M^T`` = replace each block by its mean (F2 §2).

        Implemented directly as reshape -> mean -> broadcast (no ``k`` scaling,
        since ``measure``'s ``k`` and ``adjoint``'s ``1/k`` cancel). Equals
        ``adjoint(measure(x))`` (asserted in a test).

        Args:
            x: A signal-domain tensor ``[B, H, W, C]``.

        Returns:
            The block-mean-broadcast projection, same shape as ``x``.
        """
        k, h, w, c = self.k, self.h, self.w, self.c
        blocks = keras.ops.reshape(x, (-1, h // k, k, w // k, k, c))
        mean = keras.ops.mean(blocks, axis=(2, 4), keepdims=True)
        out = keras.ops.tile(mean, (1, 1, k, 1, k, 1))
        return keras.ops.reshape(out, (-1, h, w, c))


class SpectralDeblurOperator(MeasurementOperator):
    """Unitary-DFT low-pass measurement for spectral deblurring (F2 §3).

    The measurement keeps the lowest ``keep_fraction`` of Fourier coefficients
    (a centered-square low-pass) of a UNITARY 2D DFT. Orthonormality is load
    bearing (INV-4): ``tf.signal.fft2d`` is the unnormalized DFT, so we normalize
    explicitly ``dft(x) = fft2d(x) / sqrt(H*W)`` and ``idft(X) = ifft2d(X) *
    sqrt(H*W)`` (``ifft2d`` already divides by ``H*W``). This pairing is mutually
    unitary: ``idft(dft(x)) = x`` exactly and Parseval holds (``||dft(x)|| =
    ||x||``). A 0/1 mask selecting a subset of orthonormal basis vectors keeps
    them orthonormal, so ``M`` has orthonormal columns.

    Real-signal caveat (INV-5, verified numerically): ``M`` maps a REAL signal
    to the Hermitian-symmetric masked spectra. Its adjoint back to real signal is
    ``real(idft(Lambda ⊙ m))`` — the ``real(...)`` annihilates any anti-Hermitian
    component. Consequently ``M^T M = I`` holds on the RANGE of ``M`` (valid
    measurements ``m = measure(x)``), NOT on arbitrary complex ``m``; and the
    adjointness identity uses the real-part Hermitian inner product
    ``real(<measure(x), m>) = <x, adjoint(m)>`` (holds for any complex ``m``).

    ``tf.signal.fft2d``/``ifft2d`` are raw-tf — one of the two accepted raw-tf
    exceptions in this repo (FFT/ifft) because keras.ops.fft2 exposes no
    orthonormal-norm control (F2 §3, MEMORY reference_prefer_remove_raw_tf).
    O(N log N), no dense matrix (INV-3).

    Attributes:
        h, w, c: Signal spatial / channel dims.
        keep_fraction: Fraction of frequencies kept along each axis.
    """

    def __init__(
        self,
        image_shape: ImageShape,
        keep_fraction: float = 0.1,
        c0: float = 0.0,
    ) -> None:
        """Build a centered low-pass DFT operator.

        Args:
            image_shape: Signal shape ``(H, W, C)``.
            keep_fraction: Fraction of frequencies to keep along each axis, in
                ``(0, 1]``. The centered square keeps ``|freq| <= r`` with
                ``r = int(keep_fraction * dim / 2)``; DC is always included.
            c0: Domain center for unmeasured components (default ``0.0``, D-002).

        Raises:
            ValueError: If ``keep_fraction`` is not in ``(0, 1]``.
        """
        super().__init__(c0=c0)
        if not (0.0 < keep_fraction <= 1.0):
            raise ValueError(f"keep_fraction must be in (0, 1]; got {keep_fraction}")
        h, w, c = int(image_shape[0]), int(image_shape[1]), int(image_shape[2])
        self.h, self.w, self.c = h, w, c
        self.keep_fraction = float(keep_fraction)
        self._sqrt_hw = tf.constant(math.sqrt(float(h * w)), dtype=tf.complex64)

        # Centered-square low-pass mask in fft2d's native corner-DC layout.
        # np.fft.fftfreq(n)*n gives the signed integer frequency at each index
        # (0, 1, ..., -1), symmetric about DC, so |freq| <= r keeps a proper
        # centered low-pass including DC (freq 0 -> always kept).
        r_h = int(self.keep_fraction * h / 2)
        r_w = int(self.keep_fraction * w / 2)
        freq_h = np.round(np.fft.fftfreq(h) * h).astype(int)
        freq_w = np.round(np.fft.fftfreq(w) * w).astype(int)
        mask2d = np.outer(np.abs(freq_h) <= r_h, np.abs(freq_w) <= r_w)
        mask = mask2d.astype(np.float32)[:, :, None]  # [H, W, 1]
        self._mask_c = tf.constant(mask, dtype=tf.complex64)
        logger.info(
            "SpectralDeblurOperator: %s, keep_fraction %.3f -> %d/%d freqs kept",
            (h, w, c), self.keep_fraction, int(mask.sum()), h * w,
        )

    def dft(self, x: TensorLike) -> "tf.Tensor":
        """Unitary 2D DFT ``fft2d(x) / sqrt(H*W)`` applied per channel.

        Transposes ``[B, H, W, C] -> [B, C, H, W]`` so ``fft2d`` acts on the last
        two (spatial) dims, mirroring ``fft_layers.py:98`` convention.

        Args:
            x: A real or complex signal-domain tensor ``[B, H, W, C]``.

        Returns:
            The unitary complex spectrum ``[B, H, W, C]``.
        """
        xc = tf.cast(x, tf.complex64)
        xt = tf.transpose(xc, [0, 3, 1, 2])
        spec = tf.signal.fft2d(xt)
        spec = tf.transpose(spec, [0, 2, 3, 1])
        return spec / self._sqrt_hw

    def idft(self, spec: TensorLike) -> "tf.Tensor":
        """Unitary inverse 2D DFT ``ifft2d(X) * sqrt(H*W)`` applied per channel.

        Args:
            spec: A complex spectrum ``[B, H, W, C]``.

        Returns:
            The complex inverse transform ``[B, H, W, C]`` (caller forces real).
        """
        st = tf.transpose(tf.cast(spec, tf.complex64), [0, 3, 1, 2])
        xt = tf.signal.ifft2d(st)
        xt = tf.transpose(xt, [0, 2, 3, 1])
        return xt * self._sqrt_hw

    def measure(self, x: TensorLike) -> "tf.Tensor":
        """Apply ``M^T`` = ``Lambda ⊙ dft(x)`` -> masked COMPLEX spectrum.

        Kept full-size (same-shape masked representation, zeros outside support)
        to avoid index bookkeeping (F2 §3).

        Args:
            x: A signal-domain tensor ``[B, H, W, C]``.

        Returns:
            The masked complex spectrum ``[B, H, W, C]`` (complex64).
        """
        return keras.ops.multiply(self._mask_c, self.dft(x))

    def adjoint(self, m: TensorLike) -> "tf.Tensor":
        """Apply ``M`` = ``real(idft(Lambda ⊙ m))`` (F2 §3, INV-5).

        # DECISION plan_2026-07-06_d6b88914/D-001: force real output and re-apply
        # the mask. The naive complex adjoint would keep an imaginary part; the
        # signal is real, so ``M`` (adjoint) is ``real(idft(...))``. This makes
        # ``M^T M = I`` hold ONLY on valid measurements ``m = measure(x)``
        # (Hermitian-symmetric masked spectra), NOT on arbitrary complex ``m`` —
        # verified numerically (random non-Hermitian m gives err ~2.4, not 0).
        # Do NOT "fix" a failing random-m orthonormality test by dropping real();
        # test M^T M on measure(x) instead. See decisions.md D-001, plan INV-4/5.

        Args:
            m: A masked complex spectrum ``[B, H, W, C]``.

        Returns:
            The real signal-domain reconstruction ``[B, H, W, C]``.
        """
        masked = keras.ops.multiply(self._mask_c, m)
        return tf.math.real(self.idft(masked))

    def project(self, x: TensorLike) -> "tf.Tensor":
        """Apply ``M M^T`` = ``real(idft(Lambda ⊙ dft(x)))`` — real low-pass (INV-5).

        Args:
            x: A signal-domain tensor ``[B, H, W, C]``.

        Returns:
            The real low-pass projection, same shape as ``x``.
        """
        masked = keras.ops.multiply(self._mask_c, self.dft(x))
        return tf.math.real(self.idft(masked))


class CompressiveSensingOperator(MeasurementOperator):
    """Structured fast orthonormal compressive-sensing operator (F2 §4b).

    ``M^T = subsample_n ∘ Transform ∘ SignFlip`` where each factor is
    orthonormal / self-adjoint, so ``M`` has orthonormal columns on the FULL
    signal space and ``project = M M^T`` is a genuine idempotent projector. This
    is the scalable default: unlike a dense ``[N, n]`` Gaussian ``M`` (~15 GB at
    256x256x3, INV-3), every factor is O(N) or O(N log N) and the operator holds
    only two ``[H, W, C]`` arrays (sign + subsample mask).

    Factors:

    1. **Rademacher sign flip** ``s ∈ {-1, +1}^{H,W,C}`` (per-pixel, fixed at
       construction, seedable) — a diagonal orthonormal, self-inverse operator
       that decorrelates spatial structure so a low-frequency subsample is still
       informative.
    2. **Orthonormal transform** — a separable 2D DCT-II (``norm='ortho'``,
       real, self-inverse via DCT-III / ``idct``) applied over ``(H, W)``
       per-channel when available in this TF build, else the confirmed unitary
       complex-FFT fallback (``1/sqrt(H*W)`` forward norm, reused from
       :class:`SpectralDeblurOperator`). Orthonormal on the full space either
       way. See the ``transform_kind`` attribute for the active path.
    3. **Random frequency subsample** — a fixed random 0/1 selection mask of
       size ``n = round(measurement_ratio * N)`` (built once, seedable). Kept in
       the SAME shape as the signal (zeros outside support), mirroring the
       spectral operator's masked-spectrum representation for batch-friendliness.

    * ``measure(x) = subsample(transform(sign_flip(x)))``.
    * ``adjoint(m) = sign_flip(inverse_transform(subsample(m)))`` — every factor
      run in reverse (transform orthogonal so ``transform^-1 = transform^T``,
      sign-flip self-inverse, ``subsample`` = ``mask ⊙ ·`` is the zero-fill
      adjoint of itself).
    * ``project(x) = adjoint(measure(x))`` — idempotent, O(N log N).

    Because the transform is orthonormal on the FULL space (not just its range,
    unlike the complex-spectral operator), ``M^T M = I`` holds cleanly:
    ``measure(adjoint(m)) = m`` for any ``m`` supported on the subsample mask
    (in particular every ``m = measure(x)``) — verified numerically to ~7e-7.

    ``project(ones) != 1`` in general (the sign flip of a constant field is not
    constant), so the base :meth:`init_mean`'s ``c0 * (1 - project(ones))`` term
    is genuinely load-bearing here; with ``c0 = 0`` it still vanishes, leaving
    ``init_mean = adjoint(measurements)``.

    No dense ``[N, n]``/``[N, N]`` matrix is built (INV-3): the structured
    operator is the only path. (The plan permits an optional small-tile dense
    Gaussian fallback behind an ``N <= max_dense_n`` guard; it is intentionally
    OMITTED here — the structured operator is the recommended default and scales
    to full 256x256x3, so a dense fallback would add surface for no benefit.)

    Attributes:
        h, w, c: Signal spatial / channel dims.
        measurement_ratio: Fraction ``n / N`` of coefficients kept.
        n: Number of kept coefficients (``round(measurement_ratio * N)``).
        transform_kind: ``'dct'`` (preferred) or ``'fft'`` (fallback) — the
            orthonormal transform actually active in this environment.
        sign: ``[H, W, C]`` float32 Rademacher ``{-1, +1}`` diagonal.
        subsample_mask: ``[H, W, C]`` float32 0/1 selection mask (``n`` ones).
    """

    def __init__(
        self,
        image_shape: ImageShape,
        measurement_ratio: float = 0.1,
        *,
        seed: Optional[int] = None,
        c0: float = 0.0,
    ) -> None:
        """Build a structured compressive-sensing operator.

        Args:
            image_shape: Signal shape ``(H, W, C)``.
            measurement_ratio: Fraction of transform coefficients to keep, in
                ``(0, 1]`` (default ``0.1``).
            seed: Optional seed for the Rademacher sign + subsample draw
                (reproducibility).
            c0: Domain center for unmeasured components (default ``0.0``, D-002).

        Raises:
            ValueError: If ``measurement_ratio`` is not in ``(0, 1]``.
        """
        super().__init__(c0=c0)
        if not (0.0 < measurement_ratio <= 1.0):
            raise ValueError(
                f"measurement_ratio must be in (0, 1]; got {measurement_ratio}"
            )
        h, w, c = int(image_shape[0]), int(image_shape[1]), int(image_shape[2])
        self.h, self.w, self.c = h, w, c
        n_total = h * w * c
        self.measurement_ratio = float(measurement_ratio)

        rng = np.random.default_rng(seed)
        # Rademacher {-1,+1} diagonal, one draw per pixel-channel, fixed forever.
        sign = rng.integers(0, 2, size=(h, w, c)).astype(np.float32) * 2.0 - 1.0
        self.sign: np.ndarray = sign

        # Fixed random subsample: choose n distinct coefficients out of N; a
        # same-shape 0/1 mask (INV-3: an [H,W,C] array, never [n,N]).
        n = int(round(self.measurement_ratio * n_total))
        n = max(1, min(n, n_total))
        self.n: int = n
        keep = rng.choice(n_total, size=n, replace=False)
        mask_flat = np.zeros(n_total, dtype=np.float32)
        mask_flat[keep] = 1.0
        self.subsample_mask: np.ndarray = mask_flat.reshape(h, w, c)

        # DECISION plan_2026-07-06_d6b88914/D-008: PREFER the real DCT-II
        # (norm='ortho') transform; fall back to the unitary complex-FFT path
        # ONLY when tf.signal.dct is absent/broken in this TF build (probed once
        # at import, A6/Q3). Do NOT hardcode the complex-FFT path as primary: the
        # real DCT needs no Hermitian-symmetry bookkeeping and no final real(),
        # so measure/adjoint stay real end-to-end and M^T M = I holds on the FULL
        # space (not merely M's range as with the complex-spectral operator). Do
        # NOT drop the FFT branch either — it is the confirmed-available safety
        # net if a future TF build regresses DCT. See decisions.md D-008.
        self.transform_kind: str = "dct" if _DCT_AVAILABLE else "fft"

        if self.transform_kind == "fft":
            # Complex machinery only materialized on the fallback path.
            self._sqrt_hw = tf.constant(
                math.sqrt(float(h * w)), dtype=tf.complex64
            )
            self._mask_c = tf.constant(
                self.subsample_mask, dtype=tf.complex64
            )

        logger.info(
            "CompressiveSensingOperator: %s, ratio %.3f -> %d/%d coeffs "
            "(transform=%s)",
            (h, w, c), self.measurement_ratio, n, n_total, self.transform_kind,
        )

    # ------------------------------------------------------------------
    # Orthonormal transform pair (DCT primary, FFT fallback).
    # ------------------------------------------------------------------

    def transform(self, x: TensorLike) -> "tf.Tensor":
        """Forward orthonormal transform (separable 2D DCT-II or unitary DFT).

        Args:
            x: A signal-domain tensor ``[B, H, W, C]``.

        Returns:
            The transformed tensor ``[B, H, W, C]`` (real for DCT, complex64 for
            the FFT fallback).
        """
        if self.transform_kind == "dct":
            return self._dct2d(x)
        return self._dft(x)

    def inverse_transform(self, y: TensorLike) -> "tf.Tensor":
        """Inverse orthonormal transform (DCT-III / ``idct`` or inverse DFT).

        Args:
            y: A transform-domain tensor ``[B, H, W, C]``.

        Returns:
            The inverse-transformed tensor ``[B, H, W, C]``.
        """
        if self.transform_kind == "dct":
            return self._idct2d(y)
        return self._idft(y)

    @staticmethod
    def _dct2d(x: TensorLike) -> "tf.Tensor":
        """Separable orthonormal 2D DCT-II over ``(H, W)``, per channel.

        ``tf.signal.dct`` only transforms the last axis, so each spatial axis is
        transposed to last, transformed, and transposed back.
        """
        xt = keras.ops.convert_to_tensor(x)
        # DCT over W (axis 2): move W to last -> [B, H, C, W].
        yw = tf.signal.dct(tf.transpose(xt, [0, 1, 3, 2]), type=2, norm="ortho")
        yw = tf.transpose(yw, [0, 1, 3, 2])
        # DCT over H (axis 1): move H to last -> [B, W, C, H].
        yh = tf.signal.dct(tf.transpose(yw, [0, 2, 3, 1]), type=2, norm="ortho")
        return tf.transpose(yh, [0, 3, 1, 2])

    @staticmethod
    def _idct2d(y: TensorLike) -> "tf.Tensor":
        """Inverse of :meth:`_dct2d` (DCT-III via ``idct``, same transposes)."""
        yt = keras.ops.convert_to_tensor(y)
        # Inverse over H first (any order works — axes are independent).
        xh = tf.signal.idct(tf.transpose(yt, [0, 2, 3, 1]), type=2, norm="ortho")
        xh = tf.transpose(xh, [0, 3, 1, 2])
        xw = tf.signal.idct(tf.transpose(xh, [0, 1, 3, 2]), type=2, norm="ortho")
        return tf.transpose(xw, [0, 1, 3, 2])

    def _dft(self, x: TensorLike) -> "tf.Tensor":
        """Unitary 2D DFT ``fft2d(x) / sqrt(H*W)`` per channel (FFT fallback)."""
        xc = tf.cast(keras.ops.convert_to_tensor(x), tf.complex64)
        spec = tf.signal.fft2d(tf.transpose(xc, [0, 3, 1, 2]))
        return tf.transpose(spec, [0, 2, 3, 1]) / self._sqrt_hw

    def _idft(self, spec: TensorLike) -> "tf.Tensor":
        """Unitary inverse 2D DFT ``ifft2d(X) * sqrt(H*W)`` (FFT fallback)."""
        st = tf.cast(keras.ops.convert_to_tensor(spec), tf.complex64)
        xt = tf.signal.ifft2d(tf.transpose(st, [0, 3, 1, 2]))
        return tf.transpose(xt, [0, 2, 3, 1]) * self._sqrt_hw

    # ------------------------------------------------------------------
    # Measurement / adjoint / projector.
    # ------------------------------------------------------------------

    def measure(self, x: TensorLike) -> "tf.Tensor":
        """Apply ``M^T`` = ``subsample(transform(sign_flip(x)))`` (F2 §4b).

        Args:
            x: A signal-domain tensor ``[B, H, W, C]``.

        Returns:
            The masked transform-domain measurement ``[B, H, W, C]`` (real for
            DCT, complex64 for the FFT fallback).
        """
        signed = keras.ops.multiply(self.sign, x)
        coeffs = self.transform(signed)
        mask = self._mask_c if self.transform_kind == "fft" else self.subsample_mask
        return keras.ops.multiply(mask, coeffs)

    def adjoint(self, m: TensorLike) -> "tf.Tensor":
        """Apply ``M`` = ``sign_flip(inverse_transform(subsample(m)))`` (F2 §4b).

        The FFT fallback forces ``tf.math.real`` after the inverse transform
        (the signal is real; the mask keeps the spectrum only approximately
        Hermitian to float precision). The DCT path is real throughout.

        Args:
            m: A masked transform-domain measurement ``[B, H, W, C]``.

        Returns:
            The real signal-domain reconstruction ``[B, H, W, C]``.
        """
        mask = self._mask_c if self.transform_kind == "fft" else self.subsample_mask
        masked = keras.ops.multiply(mask, m)
        recon = self.inverse_transform(masked)
        if self.transform_kind == "fft":
            recon = tf.math.real(recon)
        return keras.ops.multiply(self.sign, recon)
