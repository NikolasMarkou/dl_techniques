"""Variance-controlled weight initializer for Kolmogorov-Arnold Networks (KAN).

Implements the three initialization schemes of Rigas et al. (2026), "Principled
Initialization for Kolmogorov-Arnold Networks", adapted to this repository's
``KANLinear`` layer. A KAN edge applies a learnable univariate function

    f(x) = r * SiLU(x) + sum_m b_m * B_m(x)

where ``r`` is the residual (base-scaler) weight and ``b_m`` are the B-spline
coefficient weights. The two roles have very different sensitivities, so a
single Glorot/He-style scalar variance is inappropriate; the paper derives
per-role variance formulas that keep the forward/backward signal magnitude
stable across depth.

``KANInitializer`` is a single shape-driven class that produces the init tensor
for either role, selected by ``target``:

* ``target='residual'`` -> 2D ``(n_in, n_out)`` tensor for the ``base_scaler``.
* ``target='spline'``   -> 3D ``(n_in, n_out, n_coeffs)`` tensor for the
  ``spline_weight``.

Three schemes set the per-role standard deviation:

* ``'power_law'`` (paper default): ``sigma_r = (n_in * N) ** -alpha`` and
  ``sigma_b = (n_in * N) ** -beta``. With ``beta > alpha`` the residual path
  is initialized with larger magnitude than the spline path, biasing the edge
  toward its (well-conditioned) SiLU component early in training.
* ``'glorot_inspired'``: bidirectional Glorot-style variance using the expected
  squared activations / derivatives of the SiLU and basis functions
  (``mu_R_0, mu_R_1, mu_B_0, mu_B_1``), divided across the ``N`` basis count.
* ``'baseline'``: Glorot-uniform-equivalent residual std plus a small fixed
  spline noise (``baseline_noise``) — a sane control / fallback.

The basis count ``N`` used by the variance formulas is pinned to the host
layer's actual spline last-dimension ``grid_size + spline_order`` (NOT the
paper's ``G + k + 1``) for residual/spline consistency — see ``D-001``.

Reference:
    Rigas, S. et al. (2026). Principled Initialization for Kolmogorov-Arnold
    Networks.
"""

import keras
import numpy as np
from keras import ops
from typing import Any, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

_VALID_SCHEMES = ("power_law", "glorot_inspired", "baseline")
_VALID_TARGETS = ("residual", "spline")

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class KANInitializer(keras.initializers.Initializer):
    """Variance-controlled initializer for KAN residual / spline weights.

    Produces a Gaussian init tensor whose per-element standard deviation is set
    by the selected ``scheme`` and ``target`` according to the Rigas et al.
    (2026) variance formulas. Dimensions are inferred from the requested
    ``shape``: a 2D shape ``(n_in, n_out)`` is required for ``target='residual'``
    and a 3D shape ``(n_in, n_out, n_coeffs)`` for ``target='spline'``.

    Args:
        scheme: One of ``'power_law'``, ``'glorot_inspired'``, ``'baseline'``.
            Defaults to ``'power_law'``.
        target: Weight role this instance initializes — ``'residual'`` (2D
            ``base_scaler``) or ``'spline'`` (3D ``spline_weight``). Defaults to
            ``'residual'``.
        grid_size: B-spline grid size ``G`` of the host ``KANLinear``. Used to
            derive ``N`` for the 2D residual target and the basis statistics.
            Defaults to 5.
        spline_order: B-spline order ``k`` of the host ``KANLinear``. Defaults
            to 3.
        alpha: Power-law exponent for the residual std. Defaults to 0.25.
        beta: Power-law exponent for the spline std (``beta > alpha`` biases the
            edge toward the residual path). Defaults to 1.75.
        baseline_noise: Fixed spline std used by the ``'baseline'`` scheme.
            Defaults to 0.1.
        seed: Optional integer seed for reproducible sampling.

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
        alpha: float = 0.25,
        beta: float = 1.75,
        baseline_noise: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
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

        self.scheme = scheme
        self.target = target
        self.grid_size = int(grid_size)
        self.spline_order = int(spline_order)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.baseline_noise = float(baseline_noise)
        self.seed = seed

        # -------------------------------------------------------------
        # Activation / basis statistics (deterministic, numpy-only).
        # mu_R_0 = E[SiLU(x)^2], mu_R_1 = E[SiLU'(x)^2] over x ~ U(-1, 1).
        # mu_B_0 / mu_B_1 are the expected squared basis value / derivative
        # proxies used by the Glorot-inspired bidirectional variance.
        # -------------------------------------------------------------
        rng = np.random.default_rng(self.seed)
        x = rng.uniform(-1.0, 1.0, size=10000)
        sigmoid = 1.0 / (1.0 + np.exp(-x))
        r = x * sigmoid
        self.mu_R_0 = float(np.mean(r ** 2))
        r_prime = sigmoid + x * sigmoid * (1.0 - sigmoid)
        self.mu_R_1 = float(np.mean(r_prime ** 2))
        self.mu_B_0 = 1.0 / (self.grid_size + 1)
        self.mu_B_1 = 1.0

        logger.debug(
            f"Initialized KANInitializer(scheme={self.scheme}, "
            f"target={self.target}, grid_size={self.grid_size}, "
            f"spline_order={self.spline_order}, alpha={self.alpha}, "
            f"beta={self.beta}, baseline_noise={self.baseline_noise}, "
            f"seed={self.seed}); mu_R_0={self.mu_R_0:.4f}, "
            f"mu_R_1={self.mu_R_1:.4f}"
        )

    # -----------------------------------------------------------------
    # per-scheme std (each returns (sigma_r, sigma_b))
    # -----------------------------------------------------------------

    def _compute_std_power_law(
        self, n_in: int, n_out: int, N: int
    ) -> Tuple[float, float]:
        denom = n_in * N
        sigma_r = denom ** (-self.alpha)
        sigma_b = denom ** (-self.beta)
        return sigma_r, sigma_b

    def _compute_std_glorot(
        self, n_in: int, n_out: int, N: int
    ) -> Tuple[float, float]:
        var_r = (1.0 / N) * (2.0 / (n_in * self.mu_R_0 + n_out * self.mu_R_1))
        var_b = (1.0 / N) * (2.0 / (n_in * self.mu_B_0 + n_out * self.mu_B_1))
        return float(np.sqrt(var_r)), float(np.sqrt(var_b))

    def _compute_std_baseline(
        self, n_in: int, n_out: int, N: int
    ) -> Tuple[float, float]:
        glorot_limit = np.sqrt(6.0 / (n_in + n_out))
        sigma_r = float(glorot_limit / np.sqrt(3.0))
        sigma_b = self.baseline_noise
        return sigma_r, sigma_b

    # -----------------------------------------------------------------

    def __call__(
        self,
        shape: Tuple[int, ...],
        dtype: Optional[Union[str, Any]] = None,
    ) -> Any:
        if dtype is None:
            dtype = keras.backend.floatx()
        shape = tuple(int(d) for d in shape)

        if self.target == "spline":
            if len(shape) != 3:
                raise ValueError(
                    f"target='spline' requires a 3D shape "
                    f"(n_in, n_out, n_coeffs), got {shape}"
                )
            n_in, n_out = shape[0], shape[1]
            # DECISION plan_2026-06-12_6cc7c378/D-001: for the 3D spline target
            # derive N directly from the host's spline_weight last dim. This is
            # grid_size + spline_order (kan_linear.py:183), NOT the paper's
            # G + k + 1. Do NOT substitute grid_size + spline_order + 1 here:
            # it would mis-shape against KANLinear and desync the residual vs
            # spline variance scales. See decisions.md D-001.
            N = shape[-1]
        else:  # residual
            if len(shape) != 2:
                raise ValueError(
                    f"target='residual' requires a 2D shape (n_in, n_out), "
                    f"got {shape}"
                )
            n_in, n_out = shape[0], shape[1]
            # DECISION plan_2026-06-12_6cc7c378/D-001: the 2D residual target has
            # no spline axis to read N from, so reconstruct the host's basis
            # count as grid_size + spline_order (kan_linear.py:183) — matching
            # the layer's own arithmetic, NOT the paper's G + k + 1. See D-001.
            N = self.grid_size + self.spline_order

        if self.scheme == "power_law":
            sigma_r, sigma_b = self._compute_std_power_law(n_in, n_out, N)
        elif self.scheme == "glorot_inspired":
            sigma_r, sigma_b = self._compute_std_glorot(n_in, n_out, N)
        else:  # baseline
            sigma_r, sigma_b = self._compute_std_baseline(n_in, n_out, N)

        stddev = sigma_r if self.target == "residual" else sigma_b

        rng = np.random.default_rng(self.seed)
        values = (rng.standard_normal(size=shape) * stddev).astype("float32")
        return ops.cast(ops.convert_to_tensor(values), dtype)

    # -----------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "scheme": self.scheme,
            "target": self.target,
            "grid_size": self.grid_size,
            "spline_order": self.spline_order,
            "alpha": self.alpha,
            "beta": self.beta,
            "baseline_noise": self.baseline_noise,
            "seed": self.seed,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "KANInitializer":
        return cls(**config)

    def __repr__(self) -> str:
        return (
            f"KANInitializer(scheme={self.scheme!r}, target={self.target!r}, "
            f"grid_size={self.grid_size}, spline_order={self.spline_order}, "
            f"alpha={self.alpha}, beta={self.beta}, "
            f"baseline_noise={self.baseline_noise}, seed={self.seed})"
        )

# ---------------------------------------------------------------------

def create_kan_initializers(
    grid_size: int = 5,
    spline_order: int = 3,
    scheme: str = "power_law",
    alpha: float = 0.25,
    beta: float = 1.75,
    seed: Optional[int] = None,
) -> Tuple[KANInitializer, KANInitializer]:
    """Build a matched ``(residual_init, spline_init)`` pair for a ``KANLinear``.

    Both initializers share the same scheme and configuration, differing only in
    their ``target``. Wire them into a ``KANLinear`` so the residual
    (base-scaler) and spline (coefficient) weights are variance-controlled
    consistently:

        residual_init -> ``base_scaler_initializer``
        spline_init   -> ``kernel_initializer``

    The initializers are shape-driven; ``n_in``/``n_out`` are inferred at build
    time from the layer's weight shapes (no dimension arguments here).

    Args:
        grid_size: B-spline grid size ``G`` of the target ``KANLinear``.
            Defaults to 5.
        spline_order: B-spline order ``k`` of the target ``KANLinear``. Defaults
            to 3.
        scheme: Variance scheme — ``'power_law'``, ``'glorot_inspired'`` or
            ``'baseline'``. Defaults to ``'power_law'``.
        alpha: Power-law residual exponent. Defaults to 0.25.
        beta: Power-law spline exponent. Defaults to 1.75.
        seed: Optional integer seed (shared by both initializers).

    Returns:
        A tuple ``(residual_init, spline_init)`` of ``KANInitializer`` instances
        with ``target='residual'`` and ``target='spline'`` respectively.

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
    residual_init = KANInitializer(
        scheme=scheme,
        target="residual",
        grid_size=grid_size,
        spline_order=spline_order,
        alpha=alpha,
        beta=beta,
        seed=seed,
    )
    spline_init = KANInitializer(
        scheme=scheme,
        target="spline",
        grid_size=grid_size,
        spline_order=spline_order,
        alpha=alpha,
        beta=beta,
        seed=seed,
    )
    return residual_init, spline_init

# ---------------------------------------------------------------------
