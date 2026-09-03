"""
Latent-GMM point cloud registration. ``LatentGMMRegistration`` estimates the
rigid transform between two point clouds by mapping both to a shared latent
Gaussian mixture and solving the transform in closed form.

Both clouds pass through a shared autoencoder and correspondence network to get
soft assignments to GMM components. The component means and mixing weights come
from plain tensor ops, and a weighted Procrustes solve then recovers the
rotation and translation between the two sets of component means. That solve
uses `tf.linalg.svd` and `tf.linalg.diag` directly instead of `keras.ops`:
`keras.ops.svd` returns a different tuple order and a transposed third factor,
and `keras.ops.diag` does not batch. `keras.ops.det` is used, since it is a
drop-in replacement.

This model runs in float32 only. TensorFlow has no `Svd` kernel for half
precision on CPU or GPU, and the reflection correction in
`compute_rigid_transform` reads the sign of a determinant that is near zero
exactly when the rotation is near-degenerate, which half precision would make
unreliable regardless.

References:
    - Yuan et al., 2020. DeepGMR: Learning Latent Gaussian Mixture Models for
      Registration. ECCV 2020. (https://arxiv.org/abs/2008.09088)
    - Myronenko and Song, 2010. Point Set Registration: Coherent Point Drift.
      IEEE TPAMI 32(12). (https://arxiv.org/abs/0905.2635)
    - Umeyama, 1991. Least-Squares Estimation of Transformation Parameters
      Between Two Point Patterns. IEEE TPAMI 13(4).
    - Qi et al., 2017. PointNet: Deep Learning on Point Sets for 3D
      Classification and Segmentation. CVPR 2017.
      (https://arxiv.org/abs/1612.00593)
"""

import keras
import tensorflow as tf
from keras import ops
from typing import Dict, Any, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.losses.chamfer_loss import ChamferLoss
from dl_techniques.layers.geometric.point_cloud_autoencoder import (
    PointCloudAutoencoder, CorrespondenceNetwork)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.latent_gmm_registration.model")
class LatentGMMRegistration(keras.Model):
    """Semi-supervised point cloud registration via a latent GMM.

    Combines a feature-learning autoencoder with a GMM-based correspondence
    network to estimate the rigid transformation between two point clouds.

    Architecture:

    .. code-block:: text

        ┌──────────────────────────────────────────────────────────────┐
        │  Input: (Source PC, Target PC)                               │
        │         Shape: (B, N, 3) each                                │
        └────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  PointCloudAutoencoder                                       │
        │  ├─ Reconstructions: x_rec, y_rec  (B, N, 3)                 │
        │  ├─ Local Features: local_x, local_y  (B, N, F_local)        │
        │  └─ Global Features: global_x, global_y  (B, F_global)       │
        └────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  CorrespondenceNetwork (shared weights)                      │
        │  Inputs: (local_features, global_features)                   │
        │  Outputs: gamma_x, gamma_y  (B, N, K)                        │
        │  where K = num_gaussians (soft assignments)                  │
        └────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  GMM Parameter Estimation (non-trainable ops)                │
        │  ├─ Mixing coefficients: pi_x, pi_y  (B, K)                  │
        │  └─ Component means: mu_x, mu_y  (B, K, 3)                   │
        └────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  Rigid Transform Estimation (weighted Procrustes)            │
        │  ├─ Rotation: R  (B, 3, 3)                                   │
        │  └─ Translation: t  (B, 3)                                   │
        └────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  Output Dictionary                                           │
        │  ├─ reconstruction_x: (B, N, 3)                              │
        │  ├─ reconstruction_y: (B, N, 3)                              │
        │  ├─ estimated_r: (B, 3, 3)                                   │
        │  └─ estimated_t: (B, 3)                                      │
        └──────────────────────────────────────────────────────────────┘

        Legend: B=batch_size, N=num_points, K=num_gaussians, F=feature_dim

    :param num_gaussians: Number of latent GMM components. Must be positive.
    :type num_gaussians: int
    :param k_neighbors: Number of neighbors for local feature extraction. Must
        be positive.
    :type k_neighbors: int
    :param chamfer_weight: Weight of the Chamfer reconstruction loss. Defaults
        to 1.0.
    :type chamfer_weight: float
    :param transform_weight: Weight of the supervised transformation loss.
        Defaults to 1.0.
    :type transform_weight: float
    :param kwargs: Additional arguments for the ``keras.Model`` base class.

    :raises ValueError: If ``num_gaussians`` or ``k_neighbors`` is not positive,
        or if either weight is negative.

    Example:
        >>> model = LatentGMMRegistration(
        ...     num_gaussians=32,
        ...     k_neighbors=16,
        ...     chamfer_weight=1.0,
        ...     transform_weight=0.5
        ... )
        >>> source = keras.random.normal((8, 1024, 3))
        >>> target = keras.random.normal((8, 1024, 3))
        >>> outputs = model((source, target))
        >>> R_est = outputs["estimated_r"]  # Shape: (8, 3, 3)
        >>> t_est = outputs["estimated_t"]  # Shape: (8, 3)
    """

    def __init__(
            self,
            num_gaussians: int,
            k_neighbors: int,
            chamfer_weight: float = 1.0,
            transform_weight: float = 1.0,
            **kwargs: Any
    ) -> None:
        """Validate the config and create the autoencoder, correspondence network and loss.

        :raises ValueError: If ``num_gaussians`` or ``k_neighbors`` is not
            positive, or if either loss weight is negative.
        """
        super().__init__(**kwargs)

        if num_gaussians <= 0:
            raise ValueError(f"num_gaussians must be positive, got {num_gaussians}")
        if k_neighbors <= 0:
            raise ValueError(f"k_neighbors must be positive, got {k_neighbors}")
        if chamfer_weight < 0:
            raise ValueError(f"chamfer_weight must be non-negative, got {chamfer_weight}")
        if transform_weight < 0:
            raise ValueError(f"transform_weight must be non-negative, got {transform_weight}")

        self.num_gaussians = num_gaussians
        self.k_neighbors = k_neighbors
        self.chamfer_weight = chamfer_weight
        self.transform_weight = transform_weight

        self.autoencoder = PointCloudAutoencoder(
            k_neighbors=k_neighbors,
            name="autoencoder"
        )
        self.correspondence_net = CorrespondenceNetwork(
            num_gaussians=num_gaussians,
            name="correspondence_net"
        )

        self.chamfer_loss_fn = ChamferLoss(
            reduction="sum_over_batch_size",
            name="chamfer_loss"
        )

    def build(self, input_shape: Any) -> None:
        """Build the two stateful sub-layers by hand instead of tracing ``call``.

        # DECISION plan-2026-08-23T091307-9a110062/D-423: build cannot trace
        # ``call`` since it ends in ``tf.linalg.svd``, which rejects a
        # ``KerasTensor``. See decisions.md.

        ``self.autoencoder`` and ``self.correspondence_net`` are the only
        sub-layers with weights; ``compute_gmm_params`` and
        ``compute_rigid_transform`` are stateless.
        ``test_the_explicit_build_materializes_the_model.py`` pins the weight
        count against a real forward call. The batch axis is fixed at 1 since
        no weight shape depends on it.

        :param input_shape: A pair of shapes, ``(source_shape, target_shape)``,
            each ``(batch, num_points, 3)``.
        :type input_shape: Any
        """
        if self.built:
            return
        source_shape, target_shape = input_shape
        source = keras.KerasTensor((1,) + tuple(source_shape[1:]))
        target = keras.KerasTensor((1,) + tuple(target_shape[1:]))

        _, (local_x, local_y), (global_x, global_y) = self.autoencoder(
            (source, target)
        )
        self.correspondence_net((local_x, global_x))
        self.correspondence_net((local_y, global_y))

        super().build(input_shape)

    def call(
            self,
            inputs: Tuple[keras.KerasTensor, keras.KerasTensor],
            training: bool = False
    ) -> Dict[str, keras.KerasTensor]:
        """Run the autoencoder and correspondence network, then solve for the rigid transform.

        :param inputs: A ``(source_pc, target_pc)`` pair, each
            ``(batch_size, num_points, 3)``.
        :type inputs: Tuple[keras.KerasTensor, keras.KerasTensor]
        :param training: Whether the call is in training mode.
        :type training: bool
        :return: A dict with ``reconstruction_x``, ``reconstruction_y``
            (each ``(batch_size, num_points, 3)``), ``estimated_r``
            (``(batch_size, 3, 3)``) and ``estimated_t`` (``(batch_size, 3)``).
        :rtype: Dict[str, keras.KerasTensor]
        """
        source_pc, target_pc = inputs

        (x_rec, y_rec), (local_x, local_y), (global_x, global_y) = self.autoencoder(
            (source_pc, target_pc),
            training=training
        )

        # gamma[i, j, k] is the probability that point j belongs to component k.
        gamma_x = self.correspondence_net((local_x, global_x), training=training)
        gamma_y = self.correspondence_net((local_y, global_y), training=training)

        pi_x, mu_x = compute_gmm_params(source_pc, gamma_x)
        pi_y, mu_y = compute_gmm_params(target_pc, gamma_y)

        R_est, t_est = compute_rigid_transform(mu_x, pi_x, mu_y, pi_y)

        return {
            "reconstruction_x": x_rec,
            "reconstruction_y": y_rec,
            "estimated_r": R_est,
            "estimated_t": t_est
        }

    def train_step(
            self,
            data: Tuple[Tuple[keras.KerasTensor, keras.KerasTensor], Tuple[keras.KerasTensor, keras.KerasTensor]]
    ) -> Dict[str, keras.KerasTensor]:
        """Run a training step combining Chamfer and transformation loss.

        ``loss = chamfer_weight * L_chamfer + transform_weight * L_transform``,
        where ``L_chamfer`` is the summed Chamfer distance over both point
        clouds and ``L_transform = ||I - R_est^T R_gt||^2 + ||t_est - t_gt||^2``.

        :param data: A ``((source_pc, target_pc), (R_gt, t_gt))`` pair — point
            clouds each ``(batch_size, num_points, 3)``, ground-truth rotation
            ``(batch_size, 3, 3)`` and translation ``(batch_size, 3)``.
        :type data: Tuple[Tuple[keras.KerasTensor, keras.KerasTensor], Tuple[keras.KerasTensor, keras.KerasTensor]]
        :return: A dict with ``loss``, ``chamfer_loss``, ``transform_loss`` and
            any compiled metrics.
        :rtype: Dict[str, keras.KerasTensor]
        """
        (source_pc, target_pc), (R_gt, t_gt) = data

        # keras.backend has no GradientTape in Keras 3; the tape is the backend's own.
        with tf.GradientTape() as tape:
            y_pred = self((source_pc, target_pc), training=True)

            loss_chamfer_x = self.chamfer_loss_fn(source_pc, y_pred["reconstruction_x"])
            loss_chamfer_y = self.chamfer_loss_fn(target_pc, y_pred["reconstruction_y"])
            total_chamfer_loss = loss_chamfer_x + loss_chamfer_y

            R_est, t_est = y_pred["estimated_r"], y_pred["estimated_t"]

            # ||I - R_est^T R_gt||_F^2: zero when R_est and R_gt agree.
            loss_r = keras.ops.mean(
                keras.ops.square(
                    keras.ops.eye(3) - keras.ops.matmul(
                        keras.ops.transpose(R_est, (0, 2, 1)),
                        R_gt
                    )
                )
            )

            loss_t = keras.ops.mean(keras.ops.square(t_est - t_gt))

            total_transform_loss = loss_r + loss_t

            total_loss = (
                    self.chamfer_weight * total_chamfer_loss +
                    self.transform_weight * total_transform_loss
            )

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply(gradients, trainable_vars)

        # compute_metrics accepts a structured y/y_pred; compiled_metrics does not.
        metric_results = self.compute_metrics(
            x=(source_pc, target_pc),
            y={
                "reconstruction_x": source_pc,
                "reconstruction_y": target_pc,
                "estimated_r": R_gt,
                "estimated_t": t_gt,
            },
            y_pred=y_pred,
        )

        # The explicit loss keys come last: `metric_results` carries the loss
        # tracker's stale value, which would otherwise shadow this step's loss.
        return {
            **metric_results,
            "loss": total_loss,
            "chamfer_loss": total_chamfer_loss,
            "transform_loss": total_transform_loss,
        }

    def test_step(
            self,
            data: Tuple[Tuple[keras.KerasTensor, keras.KerasTensor], Tuple[keras.KerasTensor, keras.KerasTensor]]
    ) -> Dict[str, keras.KerasTensor]:
        """Run the same loss computation as :meth:`train_step`, without gradients.

        :param data: A ``((source_pc, target_pc), (R_gt, t_gt))`` pair — point
            clouds each ``(batch_size, num_points, 3)``, ground-truth rotation
            ``(batch_size, 3, 3)`` and translation ``(batch_size, 3)``.
        :type data: Tuple[Tuple[keras.KerasTensor, keras.KerasTensor], Tuple[keras.KerasTensor, keras.KerasTensor]]
        :return: A dict with ``loss``, ``chamfer_loss``, ``transform_loss`` and
            any compiled metrics.
        :rtype: Dict[str, keras.KerasTensor]
        """
        (source_pc, target_pc), (R_gt, t_gt) = data

        y_pred = self((source_pc, target_pc), training=False)

        loss_chamfer_x = self.chamfer_loss_fn(source_pc, y_pred["reconstruction_x"])
        loss_chamfer_y = self.chamfer_loss_fn(target_pc, y_pred["reconstruction_y"])
        total_chamfer_loss = loss_chamfer_x + loss_chamfer_y

        R_est, t_est = y_pred["estimated_r"], y_pred["estimated_t"]

        loss_r = keras.ops.mean(
            keras.ops.square(
                keras.ops.eye(3) - keras.ops.matmul(
                    keras.ops.transpose(R_est, (0, 2, 1)),
                    R_gt
                )
            )
        )

        loss_t = keras.ops.mean(keras.ops.square(t_est - t_gt))

        total_transform_loss = loss_r + loss_t

        total_loss = (
                self.chamfer_weight * total_chamfer_loss +
                self.transform_weight * total_transform_loss
        )

        # compute_metrics accepts a structured y/y_pred; compiled_metrics does not.
        metric_results = self.compute_metrics(
            x=(source_pc, target_pc),
            y={
                "reconstruction_x": source_pc,
                "reconstruction_y": target_pc,
                "estimated_r": R_gt,
                "estimated_t": t_gt,
            },
            y_pred=y_pred,
        )

        # The explicit loss keys come last: `metric_results` carries the loss
        # tracker's stale value, which would otherwise shadow this step's loss.
        return {
            **metric_results,
            "loss": total_loss,
            "chamfer_loss": total_chamfer_loss,
            "transform_loss": total_transform_loss,
        }

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration dictionary for serialization.

        :return: All constructor parameters needed to recreate this instance.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'num_gaussians': self.num_gaussians,
            'k_neighbors': self.k_neighbors,
            'chamfer_weight': self.chamfer_weight,
            'transform_weight': self.transform_weight
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LatentGMMRegistration":
        """Create a model from its configuration dictionary.

        :param config: Configuration dictionary from :meth:`get_config`.
        :type config: Dict[str, Any]
        :return: A new model instance.
        :rtype: LatentGMMRegistration
        """
        return cls(**config)


def compute_gmm_params(
        points: keras.KerasTensor,
        gamma: keras.KerasTensor
) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
    """Compute GMM mixing coefficients and component means from soft assignments.

    ``pi_k = mean_i gamma_ik`` and ``mu_k = sum_i(gamma_ik * x_i) / sum_i gamma_ik``.

    :param points: Point cloud coordinates.
    :type points: keras.KerasTensor, shape (batch_size, num_points, 3)
    :param gamma: Soft assignments; ``gamma[b, i, k]`` is the responsibility of
        component k for point i in batch b. Each row over k sums to 1.
    :type gamma: keras.KerasTensor, shape (batch_size, num_points, num_gaussians)
    :return: ``(pi, mu)`` — mixing coefficients
        ``(batch_size, num_gaussians)`` and component means
        ``(batch_size, num_gaussians, 3)``.
    :rtype: Tuple[keras.KerasTensor, keras.KerasTensor]
    """
    pi = keras.ops.mean(gamma, axis=1)

    gamma_expanded = keras.ops.expand_dims(gamma, axis=-1)  # (B, N, K, 1)
    points_expanded = keras.ops.expand_dims(points, axis=2)  # (B, N, 1, 3)

    weighted_sum = keras.ops.sum(
        gamma_expanded * points_expanded,
        axis=1
    )  # (B, K, 3)

    # Normalize by the total responsibility, not by `pi`: dividing by the mean
    # instead inflates every component mean, and the translation with it, by N.
    gamma_sum = keras.ops.sum(gamma, axis=1, keepdims=False)  # (B, K)
    gamma_sum_expanded = keras.ops.expand_dims(gamma_sum, axis=-1) + 1e-8  # (B, K, 1)
    mu = weighted_sum / gamma_sum_expanded  # Shape: (B, K, 3)

    return pi, mu


def compute_rigid_transform(
        mu_source: keras.KerasTensor,
        pi_source: keras.KerasTensor,
        mu_target: keras.KerasTensor,
        pi_target: keras.KerasTensor
) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
    """Solve the weighted-Procrustes rigid transform between two sets of GMM means.

    Finds the rotation R and translation t minimizing
    ``sum_k w_k * ||R mu_source_k + t - mu_target_k||^2``, where
    ``w_k = pi_source_k * pi_target_k``.

    :param mu_source: Source GMM component means.
    :type mu_source: keras.KerasTensor, shape (batch_size, num_gaussians, 3)
    :param pi_source: Source mixing coefficients.
    :type pi_source: keras.KerasTensor, shape (batch_size, num_gaussians)
    :param mu_target: Target GMM component means.
    :type mu_target: keras.KerasTensor, shape (batch_size, num_gaussians, 3)
    :param pi_target: Target mixing coefficients.
    :type pi_target: keras.KerasTensor, shape (batch_size, num_gaussians)
    :return: ``(R, t)`` — a proper rotation matrix (``det(R) = +1``),
        shape ``(batch_size, 3, 3)``, and a translation, shape
        ``(batch_size, 3)``, aligning the centroids after rotation.
    :rtype: Tuple[keras.KerasTensor, keras.KerasTensor]
    """
    weights = keras.ops.expand_dims(
        pi_source * pi_target,
        axis=-1
    )  # (B, K, 1)

    weight_sum = keras.ops.sum(weights, axis=1) + 1e-8  # (B, 1)

    centroid_source = keras.ops.sum(weights * mu_source, axis=1) / weight_sum  # (B, 3)
    centroid_target = keras.ops.sum(weights * mu_target, axis=1) / weight_sum  # (B, 3)

    mu_source_centered = mu_source - keras.ops.expand_dims(centroid_source, axis=1)
    mu_target_centered = mu_target - keras.ops.expand_dims(centroid_target, axis=1)

    # DECISION plan_2026-06-15_00924f53/D-001: broadcast weights via transpose to
    # (B, 1, K), not expand_dims(axis=1), which gives rank-4 and crashes the Mul.
    H = keras.ops.matmul(
        keras.ops.transpose(mu_source_centered, (0, 2, 1)) * keras.ops.transpose(weights, (0, 2, 1)),
        mu_target_centered
    )  # (B, 3, 3)

    # tf.linalg.svd is kept for its (s, u, v) tuple order and un-transposed v,
    # which keras.ops.svd does not match. See the module docstring.
    # DECISION plan_2026-06-15_00924f53/D-001: bind u, v as (s, u, v), not
    # (u, s, v); the prior unpack mis-bound s to U and crashed the transpose.
    # DECISION plan-2026-08-19T163559-499b6f0e/D-011: do not wrap this in a
    # float32 cast island for mixed_float16; TF has no half-precision Svd kernel.
    _s, U, V = tf.linalg.svd(H)

    R = keras.ops.matmul(V, keras.ops.transpose(U, (0, 2, 1)))

    # DECISION plan-2026-08-19T163559-499b6f0e/D-083: keras.ops.det is batched
    # and a measured drop-in here; keras.ops.svd and keras.ops.diag are not.
    det = ops.det(R)  # (B,)

    # diag([1, 1, det(R)]) is identity when det(R) = +1, else flips the third axis.
    correction = keras.ops.stack([
        keras.ops.ones_like(det),
        keras.ops.ones_like(det),
        det
    ], axis=-1)  # (B, 3)
    correction_matrix = tf.linalg.diag(correction)  # (B, 3, 3)

    R = keras.ops.matmul(
        keras.ops.matmul(V, correction_matrix),
        keras.ops.transpose(U, (0, 2, 1))
    )

    t = centroid_target - keras.ops.squeeze(
        keras.ops.matmul(R, keras.ops.expand_dims(centroid_source, axis=-1)),
        axis=-1
    )  # (B, 3)

    return R, t

# ---------------------------------------------------------------------


def create_latent_gmm_registration(
        num_gaussians: int = 32,
        k_neighbors: int = 16,
        chamfer_weight: float = 1.0,
        transform_weight: float = 1.0,
        **kwargs: Any
) -> LatentGMMRegistration:
    """Create a LatentGMMRegistration model.

    There is no ``MODEL_VARIANTS`` table for this architecture: the paper
    defines a single network and only ``num_gaussians`` / ``k_neighbors``
    scale it, so this factory constructs the class with the paper defaults
    rather than delegating to ``from_variant``.

    :param num_gaussians: Number of latent GMM components. Must be positive.
    :type num_gaussians: int
    :param k_neighbors: Number of neighbors for local feature extraction. Must
        be positive and smaller than the number of input points.
    :type k_neighbors: int
    :param chamfer_weight: Weight of the Chamfer reconstruction loss.
    :type chamfer_weight: float
    :param transform_weight: Weight of the supervised transformation loss.
    :type transform_weight: float
    :param kwargs: Additional arguments forwarded to the model constructor.

    :return: A configured ``LatentGMMRegistration`` instance.
    :rtype: LatentGMMRegistration
    :raises ValueError: If any numeric argument is out of range.

    Example:
        >>> model = create_latent_gmm_registration(num_gaussians=8, k_neighbors=8)
        >>> out = model((keras.random.normal((2, 64, 3)),
        ...              keras.random.normal((2, 64, 3))))
        >>> tuple(out["estimated_r"].shape)
        (2, 3, 3)
    """
    return LatentGMMRegistration(
        num_gaussians=num_gaussians,
        k_neighbors=k_neighbors,
        chamfer_weight=chamfer_weight,
        transform_weight=transform_weight,
        **kwargs
    )

# ---------------------------------------------------------------------
