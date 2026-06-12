# DECISION plan_2026-06-11_f662207d/D-010
# THERA aliasing TV penalty = mean(abs(jacobian)) where `jacobian` is the EXACT
# analytic spatial Jacobian d(field)/d(rel_coords) at t=0 produced by
# `Thera(..., return_jac=True)` / `TheraHypernetwork.decode_with_jac` (NOT a
# finite-difference proxy -- Q3 forbids that; see decisions.md D-010). This module
# is a pure-function consumer of that Jacobian; the exact-Jacobian construction
# (nested tf.GradientTape.batch_jacobian, second-order-safe) lives in
# `models/thera/hypernetwork.py`. Keep this an L1 (abs) reduction to match THERA's
# reference `utils.compute_metrics` -> `metrics['TV'] = mean(abs(jacobian))`.
"""THERA exact Jacobian-TV aliasing regularizer (loss helpers).

THERA's headline contribution is an *aliasing-free* arbitrary-scale super-
resolution field. The training objective is::

    total = recon + tv_weight * TV
    TV    = mean(|d(field)/d(rel_coords)|)        # at t = 0 (the un-smoothed field)

where ``recon`` is the per-pixel reconstruction term (MAE / Charbonnier, computed
by the trainer on the *denormalized* output) and ``TV`` is the L1 total-variation
of the EXACT per-pixel spatial Jacobian of the neural heat field with respect to
its local query coordinate, evaluated at heat-time ``t = 0`` (envelope == 1).

The Jacobian itself is produced upstream by
:meth:`dl_techniques.models.thera.model.Thera.call` (``return_jac=True``) /
:meth:`dl_techniques.models.thera.hypernetwork.TheraHypernetwork.decode_with_jac`
as an exact nested-tape ``batch_jacobian`` -- this module only reduces it to the
scalar penalty and (optionally) sums it with the reconstruction term.

This matches the reference (``model/thera.py apply_decoder`` ``return_jac`` branch
+ ``utils.compute_metrics`` ``metrics['TV'] = mean(abs(jacobian))``).

Reference:
    Becker et al., "Thera: Aliasing-Free Arbitrary-Scale Super-Resolution with
    Neural Heat Fields".
"""

from typing import Any

import keras

# ---------------------------------------------------------------------


def thera_tv_penalty(jacobian: Any) -> Any:
    """L1 total-variation aliasing penalty of the spatial Jacobian.

    Computes ``mean(|jacobian|)``, matching THERA's reference
    ``metrics['TV'] = mean(abs(jacobian))``. The ``jacobian`` argument is the
    exact per-pixel spatial Jacobian ``d(field)/d(rel_coords)`` at ``t = 0``,
    typically shape ``(B, Hq, Wq, out_dim, 2)`` from
    ``Thera(..., return_jac=True)``, though any shape is accepted (the reduction
    is over all elements).

    Args:
        jacobian: The exact spatial Jacobian tensor (any shape). Differentiable
            w.r.t. the model weights (the upstream nested tape is second-order
            safe), so this penalty can be used directly inside the trainer's
            outer ``GradientTape``.

    Returns:
        A scalar tensor: the mean absolute Jacobian.
    """
    return keras.ops.mean(keras.ops.abs(jacobian))


def thera_total_loss(
    recon_loss_value: Any,
    jacobian: Any,
    tv_weight: float,
) -> Any:
    """THERA training objective: ``recon + tv_weight * TV``.

    Convenience combiner for the trainer's ``train_step``. ``recon_loss_value``
    is the *already-computed* reconstruction scalar (MAE / Charbonnier on the
    denormalized output -- the trainer owns the data statistics and the
    nearest-source residual add, so it computes ``recon`` itself), and
    ``jacobian`` is the exact spatial Jacobian from ``return_jac=True``.

    Args:
        recon_loss_value: Scalar reconstruction loss (e.g. MAE / Charbonnier).
        jacobian: Exact spatial Jacobian tensor (see :func:`thera_tv_penalty`).
        tv_weight: Non-negative weight on the aliasing TV term.

    Returns:
        A scalar tensor: ``recon_loss_value + tv_weight * mean(|jacobian|)``.
    """
    tv = thera_tv_penalty(jacobian)
    return recon_loss_value + tv_weight * tv

# ---------------------------------------------------------------------
