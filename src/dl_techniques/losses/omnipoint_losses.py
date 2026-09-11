"""
Decoupled ray/distance/scale losses for the OmniPoint metric point-cloud model.

Conceptual Overview:
    OmniPoint predicts a per-pixel metric 3D point as ``P = s_hat * d_hat * r_hat``, where
    ``r_hat`` is a unit ray direction, ``d_hat`` a positive radial distance, and ``s_hat`` a
    single per-sample global metric scale. Supervising ``P`` directly with a single regression
    loss couples ray-direction error and distance error together: a small ray error at a large
    distance produces the same point-space residual as a large ray error at a small distance,
    so gradients cannot tell the two apart.

    This module implements the paper's decoupling: ``L_ray`` supervises the ray direction alone
    (angle-only, scale-free), and ``L_point`` supervises the predicted *distance* alone by
    projecting it onto the GROUND-TRUTH ray (never the predicted ray) -- so a wrong predicted
    ray direction cannot leak into the distance term. ``L_metric`` separately supervises the
    global scale ``s_hat`` in log-space against a robust optimal-alignment scale ``s*`` computed
    once per batch (stop-gradient applied to the ``s*`` side only, since ``s*`` is a target, not
    a differentiable path back to the ground truth).

Mathematical Formulation:
    Given predicted ray ``r_hat``, predicted distance ``d_hat``, predicted scale ``s_hat``,
    ground-truth ray ``r_gt``, ground-truth distance ``d_gt``, and the externally-supplied
    optimal alignment scale ``s*`` (see :func:`compute_optimal_scale`):

        L_ray    = mean(|r_hat - r_gt|)
        L_point  = mean(|s* * d_hat * r_gt - d_gt * r_gt|)
        L_metric = (log(s_hat) - stop_gradient(log(s*)))^2

    ``L_point`` isolates distance error: if ``d_hat == d_gt`` exactly (correct distance) but
    ``r_hat != r_gt`` (wrong ray direction), ``L_point`` is exactly zero because it never reads
    ``r_hat`` -- this is the central decoupling claim this module's tests verify directly.

References:
    Ye et al., "OmniPoint: Universal Monocular Metric Pointcloud from Any Camera."
    See `plans/plan-2026-09-11T050223-1b47bcf6/decisions.md` D-004 for the documented
    simplification of the normal-consistency / local-consistency terms (finite-difference
    cross-product and discrete-Laplacian approximations, not PCA-based estimators).
"""

import keras
from typing import Any, Optional, Sequence

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


def _masked_mean_over_all_but_batch(
        values: keras.KerasTensor,
        valid_mask: Optional[keras.KerasTensor] = None,
        epsilon: float = 1e-8,
) -> keras.KerasTensor:
    """Reduce a per-pixel tensor to a per-sample scalar, honoring an optional validity mask.

    Args:
        values: Tensor of shape ``(batch, ...)`` -- already channel-reduced per-pixel error
            (e.g. mean absolute error over the last axis, so `values` carries no channel axis).
        valid_mask: Optional tensor broadcastable to ``values``, either the same rank
            (``(batch, ...)``) or one rank higher with a trailing size-1 channel axis
            (``(batch, ..., 1)``, the convention used for point/ray tensors before their own
            channel-axis reduction) -- the trailing size-1 axis is squeezed automatically.
            1.0 for valid pixels, 0.0 for invalid/masked-out pixels (e.g. zero-depth GT). If
            ``None``, every pixel is treated as valid.
        epsilon: Floor on the valid-pixel count denominator, preventing a divide-by-zero
            when every pixel in a sample is masked out (Problem Statement edge case:
            "GT ray+distance conversion must propagate an explicit invalid mask").

    Returns:
        Tensor of shape ``(batch,)``.
    """
    batch_size = keras.ops.shape(values)[0]
    values_flat = keras.ops.reshape(values, (batch_size, -1))

    if valid_mask is None:
        return keras.ops.mean(values_flat, axis=-1)

    mask = keras.ops.cast(valid_mask, values.dtype)
    if len(mask.shape) == len(values.shape) + 1 and mask.shape[-1] == 1:
        mask = keras.ops.squeeze(mask, axis=-1)
    mask_flat = keras.ops.reshape(
        keras.ops.broadcast_to(mask, keras.ops.shape(values)),
        (batch_size, -1),
    )
    weighted_sum = keras.ops.sum(values_flat * mask_flat, axis=-1)
    valid_count = keras.ops.maximum(keras.ops.sum(mask_flat, axis=-1), keras.ops.cast(epsilon, values.dtype))
    return weighted_sum / valid_count


def compute_optimal_scale(
        pred_affine_points: keras.KerasTensor,
        gt_points: keras.KerasTensor,
        valid_mask: Optional[keras.KerasTensor] = None,
        epsilon: float = 1e-6,
) -> keras.KerasTensor:
    """Compute the optimal scalar ``s*`` aligning ``s* * pred_affine_points`` to ``gt_points``.

    This is the standalone scale-alignment helper Steps ``PointDistanceLoss``,
    ``MetricScaleLoss`` and ``OmniPointCombinedLoss`` all consume -- ``s*`` is computed ONCE
    per batch and shared, never recomputed per loss term. It adapts the robust
    epsilon-floored-denominator pattern from
    :class:`~dl_techniques.losses.affine_invariant_loss.AffineInvariantLoss` (which floors its
    MAD-based scale denominator the same way) to the least-squares scalar-alignment problem:
    minimizing ``sum(|s * pred - gt|^2)`` over a scalar ``s`` has the closed form
    ``s* = sum(pred . gt) / sum(pred . pred)``, the per-sample dot product of the (masked,
    flattened) point clouds divided by the squared norm of the predicted point cloud.

    Args:
        pred_affine_points: Predicted UNSCALED (affine, i.e. ``d_hat * r_hat``) point cloud,
            shape ``(batch, ..., 3)``.
        gt_points: Ground-truth metric point cloud, shape ``(batch, ..., 3)``, same leading
            shape as `pred_affine_points`.
        valid_mask: Optional tensor broadcastable to `pred_affine_points`'s shape minus the
            channel axis (e.g. ``(batch, ..., 1)``), 1.0 for valid pixels (e.g. non-zero GT
            depth), 0.0 for invalid ones. Invalid pixels are excluded from the alignment sum.
            If ``None``, every pixel is treated as valid.
        epsilon: Floor on the denominator (`sum(pred . pred)`), preventing division by a
            near-zero norm for a degenerate/near-zero-scale GT point cloud (Problem Statement
            edge case). Must be positive.

    Returns:
        Tensor of shape ``(batch,)`` -- one optimal scale ``s*`` per sample. Finite even when
        `valid_mask` zeroes out an entire sample (returns 0.0 for that sample, not NaN/Inf).
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")

    batch_size = keras.ops.shape(pred_affine_points)[0]
    pred_flat = keras.ops.reshape(pred_affine_points, (batch_size, -1))
    gt_flat = keras.ops.reshape(gt_points, (batch_size, -1))

    if valid_mask is not None:
        mask_broadcast = keras.ops.broadcast_to(
            keras.ops.cast(valid_mask, pred_affine_points.dtype),
            keras.ops.shape(pred_affine_points),
        )
        mask_flat = keras.ops.reshape(mask_broadcast, (batch_size, -1))
        pred_flat = pred_flat * mask_flat
        gt_flat = gt_flat * mask_flat

    numerator = keras.ops.sum(pred_flat * gt_flat, axis=-1)
    denominator = keras.ops.sum(pred_flat * pred_flat, axis=-1)
    denominator_stable = keras.ops.maximum(denominator, keras.ops.cast(epsilon, denominator.dtype))

    return numerator / denominator_stable


@register_dl_technique("dl_techniques.losses.omnipoint_losses.ray_direction_loss")
class RayDirectionLoss(keras.losses.Loss):
    """L1 loss between predicted and ground-truth unit ray directions.

    Simple, scale-free -- no distance or scale term is involved. Used standalone via
    ``call(y_true=gt_ray, y_pred=pred_ray)`` matching the standard Keras ``Loss`` convention.

    Args:
        name: String, name of the loss function. Defaults to ``'ray_direction_loss'``.
        **kwargs: Additional keyword arguments passed to the parent `Loss` class.
    """

    def __init__(self, name: str = "ray_direction_loss", **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)

    def call(
            self,
            y_true: keras.KerasTensor,
            y_pred: keras.KerasTensor,
            valid_mask: Optional[keras.KerasTensor] = None,
    ) -> keras.KerasTensor:
        """Compute the ray-direction L1 loss.

        Args:
            y_true: Ground-truth unit ray directions, shape ``(batch, ..., 3)``.
            y_pred: Predicted unit ray directions, shape ``(batch, ..., 3)``.
            valid_mask: Optional per-pixel validity mask (see
                :func:`_masked_mean_over_all_but_batch`); NOT forwarded by Keras's
                ``Loss.__call__`` (which only passes `y_true`/`y_pred`) -- pass explicitly when
                calling `.call()` directly, as `OmniPointCombinedLoss` does.

        Returns:
            Per-sample loss, shape ``(batch,)``.
        """
        per_pixel = keras.ops.mean(keras.ops.abs(y_true - y_pred), axis=-1)
        return _masked_mean_over_all_but_batch(per_pixel, valid_mask)

    def get_config(self) -> dict[str, Any]:
        return super().get_config()


@register_dl_technique("dl_techniques.losses.omnipoint_losses.point_distance_loss")
class PointDistanceLoss(keras.losses.Loss):
    """Decoupled distance loss: predicted distance projected along the GROUND-TRUTH ray.

    ``L_point = mean(|s* * d_hat * r_gt - d_gt * r_gt|)`` -- critically, this combines the
    predicted distance ``d_hat`` with the GROUND-TRUTH ray ``r_gt`` (never the predicted ray),
    which is the paper's decoupling trick: a wrong predicted ray direction cannot contaminate
    the distance-error signal.

    This loss needs three tensors (predicted distance, GT ray, GT distance) plus an externally
    -computed scalar ``s*`` (see :func:`compute_optimal_scale`), which does not fit the
    standard two-argument ``call(y_true, y_pred)`` `Loss` contract. It is therefore called
    directly via `.call(y_true, y_pred, s_star=...)` (bypassing `Loss.__call__`'s
    `y_true`/`y_pred`-only dispatch) -- `OmniPointCombinedLoss` is the single entry point meant
    for `model.compile(loss=...)`; this class is an internal composition unit.

    Args:
        name: String, name of the loss function. Defaults to ``'point_distance_loss'``.
        **kwargs: Additional keyword arguments passed to the parent `Loss` class.
    """

    def __init__(self, name: str = "point_distance_loss", **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)

    def call(
            self,
            y_true: keras.KerasTensor,
            y_pred: keras.KerasTensor,
            s_star: Optional[keras.KerasTensor] = None,
            valid_mask: Optional[keras.KerasTensor] = None,
    ) -> keras.KerasTensor:
        """Compute the GT-ray-projected distance loss.

        Args:
            y_true: Concatenation of ``[gt_ray, gt_distance]`` along the last axis, shape
                ``(batch, ..., 4)`` (3 ray channels + 1 distance channel).
            y_pred: Predicted UNSCALED distance ``d_hat``, shape ``(batch, ..., 1)``.
            s_star: Externally-computed optimal scale, shape ``(batch,)`` or scalar-broadcastable
                to `y_pred`'s leading batch axis. Required -- raises if omitted, since this loss
                has no independent way to derive it.
            valid_mask: Optional per-pixel validity mask, see
                :func:`_masked_mean_over_all_but_batch`.

        Returns:
            Per-sample loss, shape ``(batch,)``.

        Raises:
            ValueError: If `s_star` is not supplied.
        """
        if s_star is None:
            raise ValueError(
                "PointDistanceLoss.call() requires `s_star` (the externally-computed optimal "
                "alignment scale, see compute_optimal_scale()) -- this loss does not derive it "
                "independently."
            )

        gt_ray = y_true[..., :3]
        gt_distance = y_true[..., 3:4]
        d_hat = y_pred

        s_star_bcast = keras.ops.reshape(s_star, (-1,) + (1,) * (len(gt_ray.shape) - 1))
        diff_vec = (s_star_bcast * d_hat - gt_distance) * gt_ray
        per_pixel = keras.ops.mean(keras.ops.abs(diff_vec), axis=-1)
        return _masked_mean_over_all_but_batch(per_pixel, valid_mask)

    def get_config(self) -> dict[str, Any]:
        return super().get_config()


@register_dl_technique("dl_techniques.losses.omnipoint_losses.metric_scale_loss")
class MetricScaleLoss(keras.losses.Loss):
    """Log-space, stop-gradient scale loss between predicted and optimal-aligned scale.

    ``L_metric = (log(s_hat) - stop_gradient(log(s*)))^2`` -- the stop-gradient is applied to
    the `s*` (target) side only, so gradients flow into `s_hat` (the model's own predicted
    scale) but never back into whatever tensor `s*` was derived from.

    Args:
        epsilon: Float, floor applied inside `log()` to keep both `s_hat` and `s*` away from
            zero (Problem Statement edge case: near-zero-scale GT point clouds must not produce
            a NaN/Inf loss). Defaults to 1e-6.
        name: String, name of the loss function. Defaults to ``'metric_scale_loss'``.
        **kwargs: Additional keyword arguments passed to the parent `Loss` class.
    """

    def __init__(
            self,
            epsilon: float = 1e-6,
            name: str = "metric_scale_loss",
            **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        self.epsilon = epsilon

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """Compute the log-space scale loss.

        Args:
            y_true: The optimal alignment scale ``s*``, shape ``(batch,)`` -- treated as the
                target (stop-gradient applied here).
            y_pred: The model's predicted scale ``s_hat``, shape ``(batch,)``.

        Returns:
            Per-sample loss, shape ``(batch,)``.
        """
        eps = keras.ops.cast(self.epsilon, y_true.dtype)
        log_s_star = keras.ops.stop_gradient(keras.ops.log(keras.ops.maximum(y_true, eps)))
        log_s_hat = keras.ops.log(keras.ops.maximum(y_pred, eps))
        return keras.ops.square(log_s_hat - log_s_star)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config


@register_dl_technique("dl_techniques.losses.omnipoint_losses.mask_loss")
class MaskLoss(keras.losses.Loss):
    """Binary cross-entropy loss for a per-pixel sky/validity mask.

    Thin wrapper around `keras.losses.BinaryCrossentropy(from_logits=True)`.

    Args:
        name: String, name of the loss function. Defaults to ``'mask_loss'``.
        **kwargs: Additional keyword arguments passed to the parent `Loss` class.
    """

    def __init__(self, name: str = "mask_loss", **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)
        self._bce = keras.losses.BinaryCrossentropy(
            from_logits=True, reduction=None,
        )

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """Compute the per-pixel mask BCE loss.

        Args:
            y_true: Ground-truth binary mask, shape ``(batch, ..., 1)``, values in ``{0, 1}``.
            y_pred: Predicted mask logits (pre-sigmoid), shape ``(batch, ..., 1)``.

        Returns:
            Per-sample loss, shape ``(batch,)``.
        """
        per_pixel = self._bce(y_true, y_pred)
        batch_size = keras.ops.shape(per_pixel)[0]
        return keras.ops.mean(keras.ops.reshape(per_pixel, (batch_size, -1)), axis=-1)

    def get_config(self) -> dict[str, Any]:
        return super().get_config()


# DECISION plan-2026-09-11T050223-1b47bcf6/D-004
# Do NOT replace this with a per-pixel PCA/eigendecomposition normal estimator -- that is the
# paper's actual mechanism, but it was deliberately rejected here as disproportionately
# expensive at ViT-Large's patch-grid resolution on this plan's single/dual consumer-GPU
# budget (no paper-scale cluster). This finite-difference cross-product approximation is
# the accepted, documented trade: cheaper and directly unit-testable against hand-derived
# toy tensors (Success Criterion 5), at the cost of noisier normals at depth discontinuities.
# See decisions.md D-004 for the full trade-off.
def _finite_difference_normals(
        point_map: keras.KerasTensor,
        epsilon: float = 1e-8,
) -> keras.KerasTensor:
    """Approximate per-pixel surface normals via finite-difference cross product.

    Per D-004 (a documented simplification, not a PCA/eigendecomposition-based estimator):
    ``normal ~= normalize(cross(dP/dx, dP/dy))`` using simple pixel-neighbor differences. The
    last row and column (which have no forward difference) are cropped so both partial
    derivatives share a common spatial region before the cross product.

    Args:
        point_map: Point map, shape ``(batch, height, width, 3)``.
        epsilon: Floor on the cross-product norm before normalizing, preventing division by
            zero for a degenerate (collinear-gradient) region.

    Returns:
        Tensor of shape ``(batch, height - 1, width - 1, 3)`` -- unit surface normals.
    """
    d_dx = point_map[:, :, 1:, :] - point_map[:, :, :-1, :]  # (B, H, W-1, 3)
    d_dy = point_map[:, 1:, :, :] - point_map[:, :-1, :, :]  # (B, H-1, W, 3)

    d_dx_cropped = d_dx[:, :-1, :, :]  # (B, H-1, W-1, 3)
    d_dy_cropped = d_dy[:, :, :-1, :]  # (B, H-1, W-1, 3)

    normal = keras.ops.cross(d_dx_cropped, d_dy_cropped)
    norm = keras.ops.sqrt(keras.ops.sum(keras.ops.square(normal), axis=-1, keepdims=True))
    return normal / keras.ops.maximum(norm, epsilon)


# DECISION plan-2026-09-11T050223-1b47bcf6/D-004
# Do NOT replace this with a learned/windowed local-consistency mechanism or an unsupervised
# smoothness prior -- D-004 specifically requires comparing the predicted map's discrete
# Laplacian against the GT's OWN discrete Laplacian (a supervised comparison against GT
# structure), not a self-smoothness penalty on the prediction alone. See decisions.md D-004.
def _discrete_laplacian(point_map: keras.KerasTensor) -> keras.KerasTensor:
    """Compute a discrete Laplacian: center pixel minus the mean of its 4 axis-neighbors.

    Per D-004 (a documented simplification): a simple neighbor-difference formulation, not a
    learned/windowed local-consistency mechanism.

    Args:
        point_map: Point map, shape ``(batch, height, width, 3)``.

    Returns:
        Tensor of shape ``(batch, height - 2, width - 2, 3)`` -- the interior region where all
        4 neighbors exist.
    """
    center = point_map[:, 1:-1, 1:-1, :]
    up = point_map[:, :-2, 1:-1, :]
    down = point_map[:, 2:, 1:-1, :]
    left = point_map[:, 1:-1, :-2, :]
    right = point_map[:, 1:-1, 2:, :]
    neighbor_mean = (up + down + left + right) / 4.0
    return center - neighbor_mean


@register_dl_technique("dl_techniques.losses.omnipoint_losses.normal_consistency_loss")
class NormalConsistencyLoss(keras.losses.Loss):
    """Simplified (finite-difference) surface-normal consistency loss -- see D-004.

    NOT a PCA/eigendecomposition-based normal estimator (documented, deliberate simplification
    per `decisions.md` D-004): normals are approximated as
    ``normalize(cross(dP/dx, dP/dy))`` from simple pixel-neighbor differences, then compared
    via L1 between the predicted-map-derived and GT-map-derived normal fields.

    Args:
        name: String, name of the loss function. Defaults to ``'normal_consistency_loss'``.
        **kwargs: Additional keyword arguments passed to the parent `Loss` class.
    """

    def __init__(self, name: str = "normal_consistency_loss", **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """Compute the finite-difference normal-consistency L1 loss.

        Args:
            y_true: Ground-truth point map, shape ``(batch, height, width, 3)``.
            y_pred: Predicted point map, shape ``(batch, height, width, 3)``.

        Returns:
            Per-sample loss, shape ``(batch,)``.
        """
        normal_true = _finite_difference_normals(y_true)
        normal_pred = _finite_difference_normals(y_pred)
        per_pixel = keras.ops.mean(keras.ops.abs(normal_true - normal_pred), axis=-1)
        return _masked_mean_over_all_but_batch(per_pixel)

    def get_config(self) -> dict[str, Any]:
        return super().get_config()


@register_dl_technique("dl_techniques.losses.omnipoint_losses.local_consistency_loss")
class LocalConsistencyLoss(keras.losses.Loss):
    """Simplified (discrete-Laplacian) local-consistency loss -- see D-004.

    NOT an unsupervised smoothness prior: this compares the predicted point map's discrete
    Laplacian against the GT's OWN discrete Laplacian (still a supervised comparison against GT
    structure), per D-004's exact wording.

    Args:
        name: String, name of the loss function. Defaults to ``'local_consistency_loss'``.
        **kwargs: Additional keyword arguments passed to the parent `Loss` class.
    """

    def __init__(self, name: str = "local_consistency_loss", **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """Compute the discrete-Laplacian consistency L1 loss.

        Args:
            y_true: Ground-truth point map, shape ``(batch, height, width, 3)``.
            y_pred: Predicted point map, shape ``(batch, height, width, 3)``.

        Returns:
            Per-sample loss, shape ``(batch,)``.
        """
        laplacian_true = _discrete_laplacian(y_true)
        laplacian_pred = _discrete_laplacian(y_pred)
        per_pixel = keras.ops.mean(keras.ops.abs(laplacian_true - laplacian_pred), axis=-1)
        return _masked_mean_over_all_but_batch(per_pixel)

    def get_config(self) -> dict[str, Any]:
        return super().get_config()


@register_dl_technique("dl_techniques.losses.omnipoint_losses.omnipoint_combined_loss")
class OmniPointCombinedLoss:
    """Weighted-sum combination of all OmniPoint loss terms -- the single `model.compile()` entry point.

    Implements the paper's Eq. 6 structure:
    ``L = L_point + lambda_ray*L_ray + lambda_metric*L_metric + lambda_normal*L_normal
          + lambda_local*L_local + lambda_mask*L_mask``.

    Computes the shared optimal alignment scale ``s*`` (via :func:`compute_optimal_scale`)
    exactly ONCE per call and reuses it for both `PointDistanceLoss` and `MetricScaleLoss`,
    rather than recomputing it independently per term.

    This is a plain callable (not a `keras.losses.Loss` subclass) because its `y_true`/`y_pred`
    are structured tuples of several tensors each -- Keras's `Loss.__call__` dispatch assumes a
    single tensor pair. Step 9 (`train_omnipoint.py`) resolves how `model.compile(loss=...)`
    wires a tuple-output model to this callable (see plan.md Pre-Mortem #3); this class is
    written to be callable directly as `combined_loss(y_true, y_pred)` today.

    Args:
        lambda_ray: Weight on `RayDirectionLoss`. Defaults to 1.0.
        lambda_metric: Weight on `MetricScaleLoss`. Defaults to 1.0.
        lambda_normal: Weight on `NormalConsistencyLoss`. Defaults to 1.0.
        lambda_local: Weight on `LocalConsistencyLoss`. Defaults to 1.0.
        lambda_mask: Weight on `MaskLoss`. Defaults to 1.0.
        scale_epsilon: Epsilon passed to `compute_optimal_scale`. Defaults to 1e-6.
        name: String, name of this combined loss. Defaults to ``'omnipoint_combined_loss'``.

    y_true convention: a 5-tuple
        ``(gt_ray, gt_distance, gt_point, gt_mask, valid_mask)`` where `gt_point` is the
        ground-truth metric point cloud (used for `s*` alignment and the normal/local terms)
        and `valid_mask` is broadcastable to the per-pixel spatial shape (1.0 valid, 0.0
        invalid/zero-depth).

    y_pred convention: a 4-tuple
        ``(pred_ray, pred_distance, pred_mask_logit, pred_scale)`` where `pred_distance` is the
        UNSCALED (affine) predicted distance `d_hat`, and `pred_scale` is the model's own
        predicted global scale `s_hat` (shape `(batch,)`) -- distinct from the loss-internal
        `s*` alignment target. The final metric point-cloud prediction used for the
        normal/local-consistency terms is `pred_scale * pred_ray * pred_distance`.
    """

    def __init__(
            self,
            lambda_ray: float = 1.0,
            lambda_metric: float = 1.0,
            lambda_normal: float = 1.0,
            lambda_local: float = 1.0,
            lambda_mask: float = 1.0,
            scale_epsilon: float = 1e-6,
            name: str = "omnipoint_combined_loss",
    ) -> None:
        self.lambda_ray = lambda_ray
        self.lambda_metric = lambda_metric
        self.lambda_normal = lambda_normal
        self.lambda_local = lambda_local
        self.lambda_mask = lambda_mask
        self.scale_epsilon = scale_epsilon
        self.name = name

        self._ray_loss = RayDirectionLoss()
        self._point_loss = PointDistanceLoss()
        self._metric_loss = MetricScaleLoss(epsilon=scale_epsilon)
        self._mask_loss = MaskLoss()
        self._normal_loss = NormalConsistencyLoss()
        self._local_loss = LocalConsistencyLoss()

        logger.info(
            f"Initialized OmniPointCombinedLoss with lambda_ray={lambda_ray}, "
            f"lambda_metric={lambda_metric}, lambda_normal={lambda_normal}, "
            f"lambda_local={lambda_local}, lambda_mask={lambda_mask}"
        )

    def __call__(
            self,
            y_true: Sequence[keras.KerasTensor],
            y_pred: Sequence[keras.KerasTensor],
    ) -> keras.KerasTensor:
        """Compute the full weighted-sum OmniPoint loss.

        Args:
            y_true: 5-tuple ``(gt_ray, gt_distance, gt_point, gt_mask, valid_mask)``.
            y_pred: 4-tuple ``(pred_ray, pred_distance, pred_mask_logit, pred_scale)``.

        Returns:
            Per-sample total loss, shape ``(batch,)``.
        """
        gt_ray, gt_distance, gt_point, gt_mask, valid_mask = y_true
        pred_ray, pred_distance, pred_mask_logit, pred_scale = y_pred

        # DECISION plan-2026-09-11T050223-1b47bcf6/D-026
        # Do NOT compute s* from `pred_ray * pred_distance`. The paper's own Eq. 2 defines the
        # "predicted affine-invariant point map" used for scale alignment as `d_hat * r_i` where
        # `r_i` (no hat) is the GROUND-TRUTH ray, never `r_hat` -- only L_ray ever reads the
        # predicted ray. Using `pred_ray` here lets a wrong predicted ray corrupt `s_star`, which
        # then corrupts `L_point` through this shared scale even when `pred_distance` is exactly
        # correct, silently falsifying the decoupling claim this module's own docstring and tests
        # assert. MEASURED (adversarial review, iteration-1 REFLECT, decisions.md D-026): with
        # `pred_distance == gt_distance` and a deliberately wrong `pred_ray`, the old
        # `pred_ray`-based formula gave `s_star=0.0212`, `L_point=0.665`; this formula gives
        # `s_star=1.0`, `L_point=0.0` regardless of how wrong `pred_ray` is.
        scale_alignment_points = gt_ray * pred_distance
        s_star = compute_optimal_scale(
            scale_alignment_points, gt_point, valid_mask=valid_mask, epsilon=self.scale_epsilon,
        )

        gt_ray_distance = keras.ops.concatenate([gt_ray, gt_distance], axis=-1)

        l_ray = self._ray_loss.call(gt_ray, pred_ray, valid_mask=valid_mask)
        l_point = self._point_loss.call(
            gt_ray_distance, pred_distance, s_star=s_star, valid_mask=valid_mask,
        )
        l_metric = self._metric_loss.call(s_star, pred_scale)
        l_mask = self._mask_loss.call(gt_mask, pred_mask_logit)

        # NOTE: unlike `scale_alignment_points` above, the predicted metric point cloud fed to
        # the normal/local-consistency terms legitimately uses the PREDICTED ray -- those terms
        # compare the model's own predicted geometry against GT geometry, they are not part of
        # the scale-alignment/point-loss decoupling D-026 addresses.
        pred_affine_points = pred_ray * pred_distance
        pred_metric_points = keras.ops.reshape(
            pred_scale, (-1,) + (1,) * (len(pred_affine_points.shape) - 1)
        ) * pred_affine_points
        l_normal = self._normal_loss.call(gt_point, pred_metric_points)
        l_local = self._local_loss.call(gt_point, pred_metric_points)

        total = (
                l_point
                + self.lambda_ray * l_ray
                + self.lambda_metric * l_metric
                + self.lambda_normal * l_normal
                + self.lambda_local * l_local
                + self.lambda_mask * l_mask
        )
        return total

    def get_config(self) -> dict[str, Any]:
        """Returns the configuration of this combined loss."""
        return {
            "lambda_ray": self.lambda_ray,
            "lambda_metric": self.lambda_metric,
            "lambda_normal": self.lambda_normal,
            "lambda_local": self.lambda_local,
            "lambda_mask": self.lambda_mask,
            "scale_epsilon": self.scale_epsilon,
            "name": self.name,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "OmniPointCombinedLoss":
        return cls(**config)

# ---------------------------------------------------------------------
