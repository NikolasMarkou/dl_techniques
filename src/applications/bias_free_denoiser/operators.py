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
from typing import Optional, Sequence, Tuple, Union

import keras
import numpy as np

from dl_techniques.utils.logger import logger

# Image tensors are signal-domain ``[B, H, W, C]``; ``keras.ops`` accepts either
# a NumPy array (host-side test input) or a backend tensor transparently.
TensorLike = Union[np.ndarray, "keras.KerasTensor"]

# ``(H, W, C)`` signal shape (batch excluded — operators are batch-agnostic).
ImageShape = Tuple[int, int, int]


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
