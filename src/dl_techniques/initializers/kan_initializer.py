"""Variance-controlled weight initializer for Kolmogorov-Arnold Networks (KAN).

Provides :class:`KANInitializer`, which implements the three initialization
schemes of Rigas et al., "Initialization Schemes for Kolmogorov-Arnold
Networks: An Empirical Study" (arXiv:2509.03417), adapted to this repository's
``KANLinear`` layer, and :func:`create_kan_initializers`, which builds a
matched pair for one layer.

A KAN edge applies a learnable univariate function::

    f(x) = r * SiLU(x) + sum_m b_m * B_m(x)

``r`` is the residual (base-scaler) weight and ``b_m`` are the B-spline
coefficient weights. The two roles have very different sensitivities, so a
single Glorot- or He-style scalar variance does not fit both. The paper gives a
per-role variance for each of three schemes.

:class:`KANInitializer` is one shape-driven class that produces the tensor for
either role, selected by ``target``:

* ``target='residual'`` -> 2D ``(n_in, n_out)`` tensor for the ``base_scaler``.
* ``target='spline'``   -> 3D ``(n_in, n_out, n_coeffs)`` tensor for the
  ``spline_weight``.

The three schemes
-----------------
* ``'power_law'`` is the paper's best-performing scheme: ``sigma_r =
  (n_in * N) ** -alpha`` and ``sigma_b = (n_in * N) ** -beta``. With
  ``beta > alpha`` the residual path starts larger than the spline path, biasing
  the edge toward its well-conditioned SiLU component. This is an empirical
  rule: the paper fixes ``alpha=0.25, beta=1.75`` from a grid search, not from a
  variance derivation. It is not variance preserving and it does not use
  ``n_out``, since there is no backward-path term.
* ``'glorot_inspired'`` is the paper's bidirectional Glorot-style variance,
  ``sigma^2 = (1 / N) * 2 / (n_in * mu_0 + n_out * mu_1)`` with the role's own
  expectation constants. The ``1 / N`` factor applies to both roles, as printed
  in the paper: it apportions the edge's variance budget across the edge's
  terms, which the paper writes as ``G + k + 1``, the ``G + k`` basis terms plus
  the residual term. It is not a division across ``N`` copies of a residual
  weight. This is the only scheme derived from a variance argument and the only
  width-independent one.
* ``'baseline'`` is the original KAN formulation: a Glorot-uniform-equivalent
  residual std plus a small fixed spline noise (``baseline_noise``, the paper's
  ``sigma = 0.1``). It is a control, not a variance-preserving rule.

Measured per-layer forward gain
-------------------------------
``Var(y) / Var(x)`` for one edge layer, ``x ~ U(grid_range)``, ``G=5, k=3``
(``N=8``), computed from the exact constants below. None of the three schemes is
unit gain, and this module makes no such claim:

| width | power_law (r + b) | glorot_inspired | baseline |
|-------|-------------------|-----------------|----------|
| 16    | 0.134             | 0.264           | 0.171    |
| 256   | 0.535             | 0.264           | 1.32     |
| 4096  | 2.138             | 0.264           | 19.7     |

``power_law`` with ``alpha=0.25`` scales as ``sqrt(n_in / N)`` and passes
through 1.0 only near ``n_in ~ 900`` at ``N=8``. ``alpha=0.5`` is the exponent
that would make it width-independent. ``baseline``'s fixed spline noise is
width-independent while the spline path's contribution grows as ``n_in * N``, so
its gain grows linearly with width. Call
:meth:`KANInitializer.expected_forward_gain` for a specific configuration
instead of assuming stability.

``power_law`` with ``beta=1.75`` makes the spline path numerically absent, not
merely small: its forward contribution measures 3.2e-10 at width 256 and
3.1e-13 at width 4096, so the network starts as a pure SiLU MLP to within
float32 noise. That is recoverable rather than broken, because
``dL/db_m = (dL/dy) * B_m(x)`` does not depend on ``b``, so the spline
coefficients still receive full-strength gradients from step one.

The constants
-------------
``mu_R_0 = E[SiLU(x)^2]`` and ``mu_R_1 = E[SiLU'(x)^2]`` come from a 1024-point
Gauss-Legendre quadrature over ``x ~ U(grid_range)``. ``mu_B_0 = E[B_m(x)^2]``
and ``mu_B_1 = E[B'_m(x)^2]``, averaged over the input and the basis index ``m``
as the paper defines them, are computed exactly from the host layer's own knot
vector by composite Gauss-Legendre; the integrands are piecewise polynomials, so
the rule is exact.

Do not replace these with estimates or proxies. Monte-Carlo estimates over
10,000 draws moved with the RNG seed: 7.90% spread in ``mu_R_0`` and 4.44% in
``mu_R_1`` over 200 seeds. The proxies ``1 / (G + 1)`` and ``1.0`` for
``mu_B_0`` and ``mu_B_1`` measure 2.3x to 2.8x high, and the ``mu_B_1`` proxy
ignores grid resolution entirely, where the true value grows roughly linearly in
``G``: 0.521, 1.282, 2.899 for ``G`` = 5, 10, 20 at ``k=3``.

``grid_range`` sets both the knot vector and the expectation domain. ``mu_B_0``
is invariant to it, ``mu_B_1`` scales as ``1 / width^2``, and the SiLU moments
change outright: ``(0.094493, 0.319078)`` on ``(-1, 1)`` against
``(0.467474, 0.426391)`` on ``(-2, 2)``. The default here is ``(-1, 1)``, the
paper's assumption. ``KANLinear``'s own default is ``(-2, 2)`` and the layer
does not normalize its input, so pass the host's ``grid_range`` explicitly when
it differs.

The basis count ``N`` used by the variance formulas is pinned to the host
layer's actual spline last dimension ``grid_size + spline_order``, the layer's
``num_basis_fns``, and not to the paper's ``G + k + 1``. The paper's extra
``+1`` counts the residual term alongside the ``G + k`` basis terms. The
divergence is worth ``(G + k + 1) / (G + k) = 1.125`` in variance at ``G=5,
k=3``, about 6% in standard deviation. See ``D-001``.

Reference:
    Rigas, S., Verma, D., Alexandridis, G., & Wang, Y. *Initialization Schemes
    for Kolmogorov-Arnold Networks: An Empirical Study*. arXiv:2509.03417.
"""

import keras
import numpy as np
from typing import Any, Dict, Optional, Sequence, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------
# module constants
# ---------------------------------------------------------------------

_VALID_SCHEMES = ("power_law", "glorot_inspired", "baseline")
_VALID_TARGETS = ("residual", "spline")

#: Gauss-Legendre nodes for the SiLU moments (a smooth, non-polynomial integrand).
_SILU_QUADRATURE_NODES = 1024

#: Distinct seed offsets per role, so a matched (residual, spline) pair drawn
#: from one `seed` does not share a random stream. Without them the residual
#: matrix and the leading block of the spline tensor correlate at exactly 1.0000.
_TARGET_SEED_OFFSET = {"residual": 0, "spline": 1}

# ---------------------------------------------------------------------
# exact moment computation
# ---------------------------------------------------------------------


def _silu_moments(grid_range: Tuple[float, float]) -> Tuple[float, float]:
    """Compute exact ``E[SiLU(x)^2]`` and ``E[SiLU'(x)^2]`` for ``x ~ U(grid_range)``.

    :param grid_range: ``(lo, hi)`` input domain.
    :type grid_range: tuple of float
    :return: ``(mu_R_0, mu_R_1)``. On ``(-1, 1)`` these are 0.094493 and
        0.319078.
    :rtype: tuple of float
    """
    lo, hi = grid_range
    nodes, weights = np.polynomial.legendre.leggauss(_SILU_QUADRATURE_NODES)
    x = 0.5 * (hi - lo) * nodes + 0.5 * (lo + hi)
    w = 0.5 * (hi - lo) * weights

    sigmoid = 1.0 / (1.0 + np.exp(-x))
    residual = x * sigmoid
    derivative = sigmoid + x * sigmoid * (1.0 - sigmoid)

    span = hi - lo
    return (
        float(np.sum(residual ** 2 * w) / span),
        float(np.sum(derivative ** 2 * w) / span),
    )


def _knot_vector(
    grid_size: int, spline_order: int, grid_range: Tuple[float, float]
) -> np.ndarray:
    """Build the host layer's knot vector, ``(grid_size + 2 * spline_order + 1,)``.

    Mirrors ``KANLinear``'s own grid construction: a uniform spacing
    ``h = (hi - lo) / grid_size`` extended by ``spline_order`` knots on each
    side.

    :param grid_size: ``G``, the number of interior intervals.
    :type grid_size: int
    :param spline_order: ``k``.
    :type spline_order: int
    :param grid_range: ``(lo, hi)`` knot span.
    :type grid_range: tuple of float
    :return: The knot vector.
    :rtype: numpy.ndarray
    """
    lo, hi = grid_range
    h = (hi - lo) / grid_size
    return np.arange(-spline_order, grid_size + spline_order + 1, dtype=np.float64) * h + lo


def _bspline_basis(x: np.ndarray, knots: np.ndarray, order: int) -> np.ndarray:
    """Evaluate the Cox-de Boor B-spline basis.

    :param x: Evaluation points.
    :type x: numpy.ndarray
    :param knots: The knot vector.
    :type knots: numpy.ndarray
    :param order: Spline order ``k``.
    :type order: int
    :return: Basis values, shape ``(len(x), len(knots) - order - 1)``.
    :rtype: numpy.ndarray
    """
    x = np.asarray(x, dtype=np.float64)[:, None]
    basis = ((x >= knots[:-1]) & (x < knots[1:])).astype(np.float64)
    for p in range(1, order + 1):
        basis = (
            (x - knots[:-(p + 1)]) / (knots[p:-1] - knots[:-(p + 1)]) * basis[:, :-1]
            + (knots[p + 1:] - x) / (knots[p + 1:] - knots[1:-p]) * basis[:, 1:]
        )
    return basis


def _bspline_basis_derivative(
    x: np.ndarray, knots: np.ndarray, order: int
) -> np.ndarray:
    """Evaluate the exact B-spline basis derivative via the order-lowering recurrence.

    ``B'_{i,k} = k * (B_{i,k-1} / (t_{i+k} - t_i) - B_{i+1,k-1} / (t_{i+k+1} - t_{i+1}))``
    is an identity, not a finite difference.

    :param x: Evaluation points.
    :type x: numpy.ndarray
    :param knots: The knot vector.
    :type knots: numpy.ndarray
    :param order: Spline order ``k``.
    :type order: int
    :return: Basis derivatives, same shape as :func:`_bspline_basis`.
    :rtype: numpy.ndarray
    """
    n_basis = len(knots) - order - 1
    if order == 0:
        return np.zeros((len(x), n_basis), dtype=np.float64)

    lower = _bspline_basis(x, knots, order - 1)
    return order * (
        lower[:, :-1] / (knots[order:-1] - knots[:-(order + 1)])
        - lower[:, 1:] / (knots[order + 1:] - knots[1:-order])
    )


def _basis_moments(
    grid_size: int, spline_order: int, grid_range: Tuple[float, float]
) -> Tuple[float, float]:
    """Compute exact ``E[B_m(x)^2]`` and ``E[B'_m(x)^2]``, averaged over ``x`` and ``m``.

    The integrands are piecewise polynomials of degree ``2k`` and ``2k - 2`` on
    the knot intervals, so a composite Gauss-Legendre rule with ``k + 2`` nodes
    per interval is exact rather than approximate.

    :param grid_size: ``G``.
    :type grid_size: int
    :param spline_order: ``k``.
    :type spline_order: int
    :param grid_range: ``(lo, hi)``, both the knot span and the input domain.
    :type grid_range: tuple of float
    :return: ``(mu_B_0, mu_B_1)``. At ``G=5, k=3`` on ``(-1, 1)`` these are
        0.059921 and 0.520833.
    :rtype: tuple of float
    """
    lo, hi = grid_range
    knots = _knot_vector(grid_size, spline_order, grid_range)
    n_basis = grid_size + spline_order

    nodes, weights = np.polynomial.legendre.leggauss(spline_order + 2)
    edges = np.linspace(lo, hi, grid_size + 1)

    value_energy = 0.0
    derivative_energy = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        x = 0.5 * (right - left) * nodes + 0.5 * (left + right)
        w = 0.5 * (right - left) * weights
        value_energy += float(
            np.sum(_bspline_basis(x, knots, spline_order) ** 2 * w[:, None])
        )
        derivative_energy += float(
            np.sum(
                _bspline_basis_derivative(x, knots, spline_order) ** 2 * w[:, None]
            )
        )

    span = n_basis * (hi - lo)
    return value_energy / span, derivative_energy / span

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.initializers.kan_initializer")
class KANInitializer(keras.initializers.Initializer):
    """Draw KAN residual or spline weights at a scheme-controlled variance.

    Produces a Gaussian tensor whose per-element standard deviation is set by
    the selected ``scheme`` and ``target`` according to the Rigas et al.
    variance formulas. Dimensions are inferred from the requested ``shape``: a
    2D shape ``(n_in, n_out)`` for ``target='residual'`` and a 3D shape
    ``(n_in, n_out, n_coeffs)`` for ``target='spline'``.

    **Architecture overview:**

    .. code-block:: text

        construction
              │
              ▼
        ┌──────────────────────────────────────┐
        │ _silu_moments(grid_range)            │
        │   -> mu_R_0, mu_R_1                  │
        │ _basis_moments(G, k, grid_range)     │
        │   -> mu_B_0, mu_B_1                  │
        │ exact quadrature, no RNG             │
        └──────────────────┬───────────────────┘
                           │
        __call__(shape)    ▼
        ┌──────────────────────────────────────┐
        │ target == 'residual'                 │
        │   require rank 2 (n_in, n_out)       │
        │   N = grid_size + spline_order       │
        │ target == 'spline'                   │
        │   require rank 3 (n_in,n_out,n_coef) │
        │   N = shape[-1], checked against     │
        │       grid_size + spline_order       │
        └──────────────────┬───────────────────┘
                           ▼
        ┌──────────────────────────────────────┐
        │ _stds(n_in, n_out, N)                │
        │   -> (sigma_r, sigma_b)              │
        └──────────────────┬───────────────────┘
                           │ pick sigma_r or sigma_b by target
                           ▼
        ┌──────────────────────────────────────┐
        │ keras.random.normal(0, stddev)       │
        │ seed = base seed + target offset     │
        │ compute dtype float32 or float64     │
        └──────────────────┬───────────────────┘
                           ▼
                  cast to dtype, requested shape

    **Schemes:**

    .. code-block:: text

        scheme            sigma_r                sigma_b
        ---------------   --------------------   ---------------------
        power_law         (n_in*N)^-alpha        (n_in*N)^-beta
        glorot_inspired   sqrt((1/N)*2/(n_in     sqrt((1/N)*2/(n_in
                          *mu_R_0+n_out*mu_R_1))  *mu_B_0+n_out*mu_B_1))
        baseline          sqrt(6/(n_in+n_out))   baseline_noise
                          / sqrt(3)

    None of the three is unit-gain; see the module docstring's measured table
    and :meth:`expected_forward_gain`.

    :param scheme: One of ``'power_law'``, ``'glorot_inspired'``,
        ``'baseline'``.
    :type scheme: str
    :param target: Weight role this instance initializes: ``'residual'`` for the
        2D ``base_scaler``, ``'spline'`` for the 3D ``spline_weight``.
    :type target: str
    :param grid_size: B-spline grid size ``G`` of the host ``KANLinear``. Used
        to derive ``N`` for the 2D residual target and the basis statistics.
    :type grid_size: int
    :param spline_order: B-spline order ``k`` of the host ``KANLinear``.
    :type spline_order: int
    :param grid_range: ``(lo, hi)`` knot span of the host ``KANLinear``, also
        taken as the input domain the expectation constants are computed over.
        The default ``(-1.0, 1.0)`` is the paper's assumption. ``KANLinear``'s
        own default is ``(-2.0, 2.0)`` and it does not normalize its input, so
        pass the host's value when it differs.
    :type grid_range: sequence of float
    :param alpha: Power-law exponent for the residual std. Must be finite and
        >= 0; a negative exponent inverts the law into growth with width.
    :type alpha: float
    :param beta: Power-law exponent for the spline std. Must be finite and >= 0.
        ``beta > alpha`` is what biases the edge toward the residual path, and a
        warning is logged when it does not hold.
    :type beta: float
    :param baseline_noise: Fixed spline std used by the ``'baseline'`` scheme.
        Must be positive: ``0`` gives a dead spline path and a negative value
        silently behaves as its absolute value.
    :type baseline_noise: float
    :param seed: Optional integer seed, kept verbatim on ``self.seed`` and in
        the config. Following the Keras contract an instance replays the same
        tensor at every matching shape whether or not a seed is given; a
        seedless instance resolves one from the global RNG state, so
        ``keras.utils.set_random_seed`` controls it. The actual draw seed adds a
        per-``target`` offset, so a matched ``(residual, spline)`` pair built
        from one seed does not share a random stream.
    :type seed: int or None

    :ivar mu_R_0: ``E[SiLU(x)^2]`` over ``grid_range``.
    :vartype mu_R_0: float
    :ivar mu_R_1: ``E[SiLU'(x)^2]`` over ``grid_range``.
    :vartype mu_R_1: float
    :ivar mu_B_0: ``E[B_m(x)^2]``, averaged over ``x`` and ``m``.
    :vartype mu_B_0: float
    :ivar mu_B_1: ``E[B'_m(x)^2]``, averaged over ``x`` and ``m``.
    :vartype mu_B_1: float

    :raises ValueError: If ``scheme`` or ``target`` is unknown,
        ``grid_size <= 0``, ``spline_order < 0``, ``alpha`` or ``beta`` is
        negative or non-finite, ``baseline_noise <= 0``, or ``grid_range`` is
        not an increasing finite pair.

    Example:
        >>> init = KANInitializer(scheme='power_law', target='spline',
        ...                       grid_size=5, spline_order=3, seed=0)
        >>> w = init((4, 8, 8))  # (n_in, n_out, n_coeffs); N = shape[-1] = 8

    See Also:
        ``create_kan_initializers`` for a matched ``(residual, spline)`` pair.
    """

    def __init__(
        self,
        scheme: str = "power_law",
        target: str = "residual",
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: Sequence[float] = (-1.0, 1.0),
        alpha: float = 0.25,
        beta: float = 1.75,
        baseline_noise: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        """Validate the configuration and compute the exact moment constants.

        :param scheme: One of ``'power_law'``, ``'glorot_inspired'``,
            ``'baseline'``.
        :type scheme: str
        :param target: ``'residual'`` or ``'spline'``.
        :type target: str
        :param grid_size: B-spline grid size ``G``; must be > 0.
        :type grid_size: int
        :param spline_order: B-spline order ``k``; must be >= 0.
        :type spline_order: int
        :param grid_range: Increasing finite ``(lo, hi)`` pair.
        :type grid_range: sequence of float
        :param alpha: Finite, non-negative power-law residual exponent.
        :type alpha: float
        :param beta: Finite, non-negative power-law spline exponent.
        :type beta: float
        :param baseline_noise: Positive fixed spline std for ``'baseline'``.
        :type baseline_noise: float
        :param seed: Optional integer seed.
        :type seed: int or None
        :raises ValueError: See the class docstring.
        """
        super().__init__()

        if scheme not in _VALID_SCHEMES:
            raise ValueError(
                f"scheme must be one of {_VALID_SCHEMES}, got {scheme!r}"
            )
        if target not in _VALID_TARGETS:
            raise ValueError(
                f"target must be one of {_VALID_TARGETS}, got {target!r}"
            )
        if grid_size <= 0:
            raise ValueError(f"grid_size must be > 0, got {grid_size}")
        if spline_order < 0:
            raise ValueError(f"spline_order must be >= 0, got {spline_order}")

        grid_range = tuple(float(v) for v in grid_range)
        if len(grid_range) != 2:
            raise ValueError(
                f"grid_range must be a (lo, hi) pair, got {grid_range}"
            )
        if not all(np.isfinite(v) for v in grid_range):
            raise ValueError(f"grid_range must be finite, got {grid_range}")
        if grid_range[0] >= grid_range[1]:
            raise ValueError(
                f"grid_range must satisfy lo < hi, got {grid_range}"
            )

        for name, exponent in (("alpha", alpha), ("beta", beta)):
            if not np.isfinite(exponent):
                raise ValueError(f"{name} must be finite, got {exponent}")
            if exponent < 0:
                raise ValueError(
                    f"{name} must be >= 0 -- a negative exponent inverts the "
                    f"power law into growth with width, got {exponent}"
                )
        if baseline_noise <= 0:
            raise ValueError(
                f"baseline_noise must be positive (0 gives a dead spline path, "
                f"and a negative value silently acts as its absolute value), "
                f"got {baseline_noise}"
            )

        self.scheme = scheme
        self.target = target
        self.grid_size = int(grid_size)
        self.spline_order = int(spline_order)
        self.grid_range = grid_range
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.baseline_noise = float(baseline_noise)

        if scheme == "power_law" and self.beta <= self.alpha:
            logger.warning(
                f"KANInitializer(scheme='power_law', alpha={self.alpha}, "
                f"beta={self.beta}): beta > alpha is what biases the edge "
                f"toward its residual path; with beta <= alpha the spline path "
                f"starts at or above the residual path instead"
            )

        # Mirrors keras.initializers.RandomNormal: the config keeps whatever the
        # caller passed, the resolved seed drives the draw. keras.utils.
        # set_random_seed seeds np.random, so a seedless instance stays
        # reproducible under a global seed.
        self.seed = seed
        base_seed = seed if seed is not None else int(np.random.randint(0, 2 ** 30))
        self._draw_seed = base_seed + _TARGET_SEED_OFFSET[target]

        # Exact, deterministic constants -- no Monte Carlo, no RNG dependence.
        self.mu_R_0, self.mu_R_1 = _silu_moments(self.grid_range)
        self.mu_B_0, self.mu_B_1 = _basis_moments(
            self.grid_size, self.spline_order, self.grid_range
        )

        logger.debug(
            f"Initialized KANInitializer(scheme={self.scheme}, "
            f"target={self.target}, grid_size={self.grid_size}, "
            f"spline_order={self.spline_order}, grid_range={self.grid_range}, "
            f"alpha={self.alpha}, beta={self.beta}, "
            f"baseline_noise={self.baseline_noise}, seed={self.seed}); "
            f"mu_R_0={self.mu_R_0:.6f}, mu_R_1={self.mu_R_1:.6f}, "
            f"mu_B_0={self.mu_B_0:.6f}, mu_B_1={self.mu_B_1:.6f}"
        )

    # -----------------------------------------------------------------
    # per-scheme std (each returns (sigma_r, sigma_b))
    # -----------------------------------------------------------------

    def _compute_std_power_law(
        self, n_in: int, n_out: int, N: int
    ) -> Tuple[float, float]:
        """Apply the paper's empirical power law.

        The rule has no backward-path term, unlike the other two schemes. The
        uniform signature is for dispatch; ``n_out`` does not participate.

        :param n_in: Input feature count.
        :type n_in: int
        :param n_out: Output feature count. Unused.
        :type n_out: int
        :param N: Basis count.
        :type N: int
        :return: ``(sigma_r, sigma_b)``.
        :rtype: tuple of float
        """
        # No backward-path term in this scheme.
        del n_out
        denom = n_in * N
        return denom ** (-self.alpha), denom ** (-self.beta)

    def _compute_std_glorot(
        self, n_in: int, n_out: int, N: int
    ) -> Tuple[float, float]:
        """Apply the paper's bidirectional Glorot-style variance.

        The ``1 / N`` factor is on both roles, as printed in the paper: it
        apportions the edge's variance budget across the edge's additive terms,
        which the paper's ``G + k + 1`` counts as the ``G + k`` basis terms plus
        the residual term. It is not a division across ``N`` copies of a
        residual weight.

        :param n_in: Input feature count.
        :type n_in: int
        :param n_out: Output feature count.
        :type n_out: int
        :param N: Basis count.
        :type N: int
        :return: ``(sigma_r, sigma_b)``.
        :rtype: tuple of float
        """
        var_r = (1.0 / N) * (2.0 / (n_in * self.mu_R_0 + n_out * self.mu_R_1))
        var_b = (1.0 / N) * (2.0 / (n_in * self.mu_B_0 + n_out * self.mu_B_1))
        return float(np.sqrt(var_r)), float(np.sqrt(var_b))

    def _compute_std_baseline(
        self, n_in: int, n_out: int, N: int
    ) -> Tuple[float, float]:
        """Apply the original KAN formulation: Glorot residual, fixed spline noise.

        :param n_in: Input feature count.
        :type n_in: int
        :param n_out: Output feature count.
        :type n_out: int
        :param N: Basis count. Unused.
        :type N: int
        :return: ``(sigma_r, sigma_b)``.
        :rtype: tuple of float
        """
        # The fixed noise floor ignores the basis count.
        del N
        glorot_limit = np.sqrt(6.0 / (n_in + n_out))
        return float(glorot_limit / np.sqrt(3.0)), self.baseline_noise

    def _stds(self, n_in: int, n_out: int, N: int) -> Tuple[float, float]:
        """Dispatch to the selected scheme.

        :param n_in: Input feature count.
        :type n_in: int
        :param n_out: Output feature count.
        :type n_out: int
        :param N: Basis count.
        :type N: int
        :return: ``(sigma_r, sigma_b)``.
        :rtype: tuple of float
        """
        if self.scheme == "power_law":
            return self._compute_std_power_law(n_in, n_out, N)
        if self.scheme == "glorot_inspired":
            return self._compute_std_glorot(n_in, n_out, N)
        return self._compute_std_baseline(n_in, n_out, N)

    # -----------------------------------------------------------------
    # measurable claim
    # -----------------------------------------------------------------

    def expected_forward_gain(
        self, n_in: int, n_out: int
    ) -> Tuple[float, float]:
        """Predict the per-layer forward variance gain of each path.

        ``Var(y) / Var(x) ~ n_in * sigma_r^2 * mu_R_0`` for the residual path
        and ``n_in * N * sigma_b^2 * mu_B_0`` for the spline path, with
        ``N = grid_size + spline_order``. Their sum is the layer's gain. 1.0
        would be variance preserving, and none of the three schemes achieves it
        in general; see the module docstring.

        :param n_in: Input feature count of the edge layer.
        :type n_in: int
        :param n_out: Output feature count.
        :type n_out: int
        :return: ``(residual_gain, spline_gain)``.
        :rtype: tuple of float
        """
        N = self.grid_size + self.spline_order
        sigma_r, sigma_b = self._stds(n_in, n_out, N)
        return (
            float(n_in * sigma_r ** 2 * self.mu_R_0),
            float(n_in * N * sigma_b ** 2 * self.mu_B_0),
        )

    # -----------------------------------------------------------------

    def __call__(
        self,
        shape: Tuple[int, ...],
        dtype: Optional[Union[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Draw the initialization tensor for this instance's role.

        :param shape: ``(n_in, n_out)`` for ``target='residual'``, or
            ``(n_in, n_out, n_coeffs)`` for ``target='spline'``.
        :type shape: tuple of int
        :param dtype: Data type of the result. ``None`` falls back to
            ``keras.config.floatx()``.
        :type dtype: str or None
        :param kwargs: Additional arguments (unused).
        :return: The initialized tensor.
        :rtype: tensor
        :raises ValueError: If the rank does not match ``target``, or if the
            spline last dimension disagrees with ``grid_size + spline_order``.
        """
        if dtype is None:
            dtype = keras.config.floatx()
        dtype = getattr(dtype, "name", None) or str(dtype)
        shape = tuple(int(d) for d in shape)

        if self.target == "spline":
            if len(shape) != 3:
                raise ValueError(
                    f"target='spline' requires a 3D shape "
                    f"(n_in, n_out, n_coeffs), got {shape}"
                )
            n_in, n_out = shape[0], shape[1]
            # DECISION plan_2026-06-12_6cc7c378/D-001: N is the host's spline
            # last dim, which is grid_size + spline_order (KANLinear's
            # num_basis_fns). Do NOT use the paper's G + k + 1: it mis-shapes
            # against KANLinear and desyncs the two roles' variance scales.
            # See D-001.
            N = shape[-1]
            expected_n = self.grid_size + self.spline_order
            if N != expected_n:
                raise ValueError(
                    f"spline last dim {N} != grid_size + spline_order = "
                    f"{expected_n}. The residual target reconstructs N from "
                    f"grid_size/spline_order, so the two roles' variance scales "
                    f"would silently desync (D-001). Pass the host KANLinear's "
                    f"own grid_size and spline_order."
                )
        else:  # residual
            if len(shape) != 2:
                raise ValueError(
                    f"target='residual' requires a 2D shape (n_in, n_out), "
                    f"got {shape}"
                )
            n_in, n_out = shape[0], shape[1]
            # DECISION plan_2026-06-12_6cc7c378/D-001: the 2D residual target has
            # no spline axis, so N is reconstructed as grid_size + spline_order
            # (KANLinear's num_basis_fns). Do NOT use the paper's G + k + 1.
            # See D-001.
            N = self.grid_size + self.spline_order

        sigma_r, sigma_b = self._stds(n_in, n_out, N)
        stddev = sigma_r if self.target == "residual" else sigma_b

        # keras.random rather than np.random, so the draw honours the Keras
        # global seed, runs in the resolved dtype and allocates no float64
        # numpy temporary.
        compute_dtype = dtype if dtype in ("float32", "float64") else "float32"
        values = keras.random.normal(
            shape=shape, mean=0.0, stddev=stddev,
            dtype=compute_dtype, seed=self._draw_seed,
        )
        return keras.ops.cast(values, dtype) if compute_dtype != dtype else values

    # -----------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor arguments for serialization.

        The ``seed`` written here is the one the caller passed, not the
        per-target draw seed, so a seedless initializer stays seedless across a
        round trip. The ``mu_*`` constants are omitted because they are exact
        functions of the serialized fields.

        :return: A dict holding the scheme, target, grid settings, exponents,
            ``baseline_noise`` and ``seed``.
        :rtype: dict
        """
        config = super().get_config()
        config.update({
            "scheme": self.scheme,
            "target": self.target,
            "grid_size": self.grid_size,
            "spline_order": self.spline_order,
            "grid_range": self.grid_range,
            "alpha": self.alpha,
            "beta": self.beta,
            "baseline_noise": self.baseline_noise,
            "seed": self.seed,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "KANInitializer":
        """Rebuild an initializer from a config dict.

        :param config: Configuration dictionary. A config written before
            ``grid_range`` existed omits it and picks up the default.
        :type config: dict
        :return: A new initializer.
        :rtype: KANInitializer
        """
        return cls(**config)

    def __repr__(self) -> str:
        """Return the constructor-like representation.

        :return: A string naming every configured field.
        :rtype: str
        """
        return (
            f"KANInitializer(scheme={self.scheme!r}, target={self.target!r}, "
            f"grid_size={self.grid_size}, spline_order={self.spline_order}, "
            f"grid_range={self.grid_range}, alpha={self.alpha}, "
            f"beta={self.beta}, baseline_noise={self.baseline_noise}, "
            f"seed={self.seed})"
        )

# ---------------------------------------------------------------------

def create_kan_initializers(
    grid_size: int = 5,
    spline_order: int = 3,
    scheme: str = "power_law",
    grid_range: Sequence[float] = (-1.0, 1.0),
    alpha: float = 0.25,
    beta: float = 1.75,
    baseline_noise: float = 0.1,
    seed: Optional[int] = None,
) -> Tuple[KANInitializer, KANInitializer]:
    """Build a matched ``(residual_init, spline_init)`` pair for a ``KANLinear``.

    Both initializers share the same scheme and configuration and differ only in
    their ``target``, and so in their random stream: one ``seed`` yields two
    independent draws, not two views of the same one.

    **Wiring into a KANLinear:**

    .. code-block:: text

        create_kan_initializers(G, k, scheme, ..., seed)
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
        KANInitializer              KANInitializer
        target='residual'           target='spline'
        seed + 0                    seed + 1
            │                           │
            ▼                           ▼
        base_scaler_initializer     kernel_initializer
            │                           │
            ▼                           ▼
        base_scaler                 spline_weight
        [n_in, n_out]               [n_in, n_out, G + k]

    The initializers are shape-driven: ``n_in`` and ``n_out`` are inferred at
    build time from the layer's weight shapes, so no dimension arguments are
    needed here.

    :param grid_size: B-spline grid size ``G`` of the target ``KANLinear``.
    :type grid_size: int
    :param spline_order: B-spline order ``k`` of the target ``KANLinear``.
    :type spline_order: int
    :param scheme: Variance scheme: ``'power_law'``, ``'glorot_inspired'`` or
        ``'baseline'``.
    :type scheme: str
    :param grid_range: ``(lo, hi)`` knot span of the target ``KANLinear``, also
        the domain the expectation constants are taken over. ``KANLinear``'s own
        default is ``(-2.0, 2.0)``.
    :type grid_range: sequence of float
    :param alpha: Power-law residual exponent.
    :type alpha: float
    :param beta: Power-law spline exponent.
    :type beta: float
    :param baseline_noise: Fixed spline std for the ``'baseline'`` scheme.
    :type baseline_noise: float
    :param seed: Optional integer seed, shared by both initializers, which apply
        distinct per-target offsets to it.
    :type seed: int or None
    :return: A tuple ``(residual_init, spline_init)`` of ``KANInitializer``
        instances with ``target='residual'`` and ``target='spline'``.
    :rtype: tuple of KANInitializer

    Example:
        >>> res_init, spline_init = create_kan_initializers(
        ...     grid_size=5, spline_order=3, scheme='power_law', seed=0)
        >>> from dl_techniques.layers.ffn.kan_linear import KANLinear
        >>> layer = KANLinear(
        ...     features=16,
        ...     grid_size=5,
        ...     spline_order=3,
        ...     base_scaler_initializer=res_init,
        ...     kernel_initializer=spline_init,
        ... )
    """
    shared = dict(
        scheme=scheme,
        grid_size=grid_size,
        spline_order=spline_order,
        grid_range=grid_range,
        alpha=alpha,
        beta=beta,
        baseline_noise=baseline_noise,
        seed=seed,
    )
    return (
        KANInitializer(target="residual", **shared),
        KANInitializer(target="spline", **shared),
    )

# ---------------------------------------------------------------------
