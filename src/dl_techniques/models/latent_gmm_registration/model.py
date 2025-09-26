import keras
from keras import ops
from typing import Dict, Any, Tuple

from dl_techniques.layers.geometric.point_cloud_autoencoder import PointCloudAutoencoder, CorrespondenceNetwork

def chamfer_loss(pc1: keras.KerasTensor, pc2: keras.KerasTensor) -> keras.KerasTensor:
    """Computes the Chamfer distance between two point clouds."""
    # Pairwise distance matrix: (batch, num_points1, num_points2)
    pc1_expand = ops.expand_dims(pc1, axis=2)
    pc2_expand = ops.expand_dims(pc2, axis=1)
    dist_mat = ops.sum(ops.square(pc1_expand - pc2_expand), axis=-1)

    # Minimum distances
    dist_1 = ops.min(dist_mat, axis=2)
    dist_2 = ops.min(dist_mat, axis=1)

    return ops.mean(dist_1) + ops.mean(dist_2)


@keras.saving.register_keras_serializable()
class LatentGMMRegistration(keras.Model):
    """
    Robust Semi-Supervised Point Cloud Registration via Latent GMM.

    This model implements the complete architecture from the paper, combining a
    feature-learning autoencoder with a GMM-based correspondence network to
    estimate the rigid transformation between two point clouds.

    **Intent**: To provide an end-to-end, learning-based solution for point
    cloud registration that is robust to noise and large transformations.

    **Architecture**:
    ```
    Input (Source PC, Target PC)
           |
    PointCloudAutoencoder -> Features & Reconstructions
           |
    CorrespondenceNetwork -> Point-to-GMM Mappings (gamma)
           |
    Compute GMM Params (ops) -> GMM means (mu) & mixing coeffs (pi)
           |
    Compute Rigid Transform (ops) -> Rotation (R) & Translation (t)
           |
    Output ({reconstructions, R, t})
    ```

    Args:
        num_gaussians (int): Number of latent GMM components.
        k_neighbors (int): Number of neighbors for feature extraction.
        chamfer_weight (float): Weight for the Chamfer reconstruction loss.
        transform_weight (float): Weight for the transformation loss.
        **kwargs: Additional arguments for Model base class.
    """

    def __init__(
            self,
            num_gaussians: int,
            k_neighbors: int,
            chamfer_weight: float = 1.0,
            transform_weight: float = 1.0,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.num_gaussians = num_gaussians
        self.k_neighbors = k_neighbors
        self.chamfer_weight = chamfer_weight
        self.transform_weight = transform_weight

        # CREATE all sub-layers in __init__
        self.autoencoder = PointCloudAutoencoder(k_neighbors, name="autoencoder")
        self.correspondence_net = CorrespondenceNetwork(num_gaussians, name="correspondence_net")

    def call(self, inputs: Tuple[keras.KerasTensor, keras.KerasTensor]) -> Dict[str, keras.KerasTensor]:
        source_pc, target_pc = inputs

        # 1. Feature Extraction
        (x_rec, y_rec), (local_x, local_y), (global_x, global_y) = self.autoencoder((source_pc, target_pc))

        # 2. Correspondence Estimation
        gamma_x = self.correspondence_net((local_x, global_x))
        gamma_y = self.correspondence_net((local_y, global_y))

        # 3. GMM Parameter Estimation (non-trainable ops)
        pi_x, mu_x = compute_gmm_params(source_pc, gamma_x)
        pi_y, mu_y = compute_gmm_params(target_pc, gamma_y)

        # 4. Rigid Transform Estimation (non-trainable ops)
        R_est, t_est = compute_rigid_transform(mu_x, pi_x, mu_y, pi_y)

        return {
            "reconstruction_x": x_rec,
            "reconstruction_y": y_rec,
            "estimated_r": R_est,
            "estimated_t": t_est
        }

    def train_step(self, data):
        (source_pc, target_pc), (R_gt, t_gt) = data

        with keras.backend.GradientTape() as tape:
            # Get model predictions
            y_pred = self((source_pc, target_pc), training=True)

            # Compute Chamfer loss (unsupervised)
            loss_chamfer_x = chamfer_loss(source_pc, y_pred["reconstruction_x"])
            loss_chamfer_y = chamfer_loss(target_pc, y_pred["reconstruction_y"])
            total_chamfer_loss = loss_chamfer_x + loss_chamfer_y

            # Compute transformation loss (supervised)
            R_est, t_est = y_pred["estimated_r"], y_pred["estimated_t"]
            loss_r = ops.mean(ops.square(ops.eye(3) - ops.matmul(ops.transpose(R_est, (0, 2, 1)), R_gt)))
            loss_t = ops.mean(ops.square(t_est - t_gt))
            total_transform_loss = loss_r + loss_t

            # Total weighted loss
            total_loss = (self.chamfer_weight * total_chamfer_loss +
                          self.transform_weight * total_transform_loss)

        # Compute gradients and update weights
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply(gradients, trainable_vars)

        # Update metrics
        self.compiled_metrics.update_state(
            (R_gt, t_gt),
            (y_pred["estimated_r"], y_pred["estimated_t"])
        )

        return {
            "loss": total_loss,
            "chamfer_loss": total_chamfer_loss,
            "transform_loss": total_transform_loss,
            **{m.name: m.result() for m in self.metrics},
        }

    def test_step(self, data):
        (source_pc, target_pc), (R_gt, t_gt) = data

        y_pred = self((source_pc, target_pc), training=False)

        loss_chamfer_x = chamfer_loss(source_pc, y_pred["reconstruction_x"])
        loss_chamfer_y = chamfer_loss(target_pc, y_pred["reconstruction_y"])
        total_chamfer_loss = loss_chamfer_x + loss_chamfer_y

        R_est, t_est = y_pred["estimated_r"], y_pred["estimated_t"]
        loss_r = ops.mean(ops.square(ops.eye(3) - ops.matmul(ops.transpose(R_est, (0, 2, 1)), R_gt)))
        loss_t = ops.mean(ops.square(t_est - t_gt))
        total_transform_loss = loss_r + loss_t

        total_loss = (self.chamfer_weight * total_chamfer_loss +
                      self.transform_weight * total_transform_loss)

        self.compiled_metrics.update_state(
            (R_gt, t_gt),
            (y_pred["estimated_r"], y_pred["estimated_t"])
        )

        return {
            "loss": total_loss,
            "chamfer_loss": total_chamfer_loss,
            "transform_loss": total_transform_loss,
            **{m.name: m.result() for m in self.metrics},
        }

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'num_gaussians': self.num_gaussians,
            'k_neighbors': self.k_neighbors,
            'chamfer_weight': self.chamfer_weight,
            'transform_weight': self.transform_weight
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)