"""
Training script for Matryoshka-JEPA.

Trains a Matryoshka-JEPA model on image data using multi-scale
embedding prediction with VICReg regularization.
"""

import math
from typing import Optional, Dict, Any, Tuple

import numpy as np
import tensorflow as tf
import keras
from keras import ops

from .model import (
    MatryoshkaJEPA,
    generate_batch_masks,
    cosine_ema_schedule,
)


# =============================================================================
# Data Pipeline
# =============================================================================


def create_random_image_dataset(
    batch_size: int = 32,
    image_size: int = 224,
    num_samples: int = 1000,
) -> tf.data.Dataset:
    """
    Create a dataset of random images for testing.

    In practice, replace with real image loading.

    :param batch_size: Batch size
    :type batch_size: int
    :param image_size: Image height/width
    :type image_size: int
    :param num_samples: Number of samples
    :type num_samples: int
    :return: TensorFlow dataset
    :rtype: tf.data.Dataset
    """
    def generate():
        for _ in range(num_samples):
            image = np.random.randn(image_size, image_size, 3).astype(np.float32)
            yield image

    dataset = tf.data.Dataset.from_generator(
        generate,
        output_signature=tf.TensorSpec(
            shape=(image_size, image_size, 3),
            dtype=tf.float32
        )
    )

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def create_cifar10_dataset(
    batch_size: int = 32,
    image_size: int = 224,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Create CIFAR-10 dataset for training.

    :param batch_size: Batch size
    :type batch_size: int
    :param image_size: Target image size
    :type image_size: int
    :return: Train and validation datasets
    :rtype: Tuple[tf.data.Dataset, tf.data.Dataset]
    """
    (x_train, _), (x_val, _) = keras.datasets.cifar10.load_data()

    # Normalize to [-1, 1]
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    x_val = (x_val.astype(np.float32) - 127.5) / 127.5

    def resize_and_augment(image):
        # Resize to target size
        image = tf.image.resize(image, (image_size, image_size))
        # Random horizontal flip
        image = tf.image.random_flip_left_right(image)
        return image

    def resize_only(image):
        return tf.image.resize(image, (image_size, image_size))

    train_ds = tf.data.Dataset.from_tensor_slices(x_train)
    train_ds = train_ds.shuffle(10000)
    train_ds = train_ds.map(resize_and_augment, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size, drop_remainder=True)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices(x_val)
    val_ds = val_ds.map(resize_only, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size, drop_remainder=True)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


# =============================================================================
# Training Loop
# =============================================================================


class MatryoshkaJEPATrainer:
    """
    Trainer for Matryoshka-JEPA.

    Handles the training loop including:
    - Mask generation
    - Gradient computation
    - EMA updates for target encoder
    - Logging and checkpointing

    :param model: MatryoshkaJEPA model instance
    :type model: MatryoshkaJEPA
    :param optimizer: Keras optimizer
    :type optimizer: keras.optimizers.Optimizer
    :param grid_size: Patch grid size (image_size // patch_size)
    :type grid_size: int
    :param num_targets: Number of target blocks per sample
    :type num_targets: int
    :param target_scale: Min/max target block scale
    :type target_scale: Tuple[float, float]
    """

    def __init__(
        self,
        model: MatryoshkaJEPA,
        optimizer: keras.optimizers.Optimizer,
        grid_size: int = 14,
        num_targets: int = 4,
        target_scale: Tuple[float, float] = (0.15, 0.2),
    ):
        self.model = model
        self.optimizer = optimizer
        self.grid_size = grid_size
        self.num_targets = num_targets
        self.target_scale = target_scale

        # Initialize target encoder
        self.model.initialize_target_encoder()

        # Metrics
        self.train_loss = keras.metrics.Mean(name="train_loss")
        self.pred_loss = keras.metrics.Mean(name="pred_loss")
        self.var_loss = keras.metrics.Mean(name="var_loss")
        self.cov_loss = keras.metrics.Mean(name="cov_loss")

    @tf.function
    def train_step(
        self,
        images: tf.Tensor,
        ctx_pos: tf.Tensor,
        tgt_pos: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        """
        Execute single training step.

        :param images: Batch of images [B, H, W, C]
        :type images: tf.Tensor
        :param ctx_pos: Context positions [B, N_ctx]
        :type ctx_pos: tf.Tensor
        :param tgt_pos: Target positions [B, N_tgt]
        :type tgt_pos: tf.Tensor
        :return: Dictionary of loss values
        :rtype: Dict[str, tf.Tensor]
        """
        with tf.GradientTape() as tape:
            losses = self.model.compute_jepa_loss(
                images=images,
                context_positions=ctx_pos,
                target_positions=tgt_pos,
                training=True
            )
            total_loss = losses["total_loss"]

        # Get trainable variables (context encoder + predictor)
        trainable_vars = (
            self.model.context_encoder.trainable_variables +
            self.model.predictor.trainable_variables
        )

        # Compute and apply gradients
        gradients = tape.gradient(total_loss, trainable_vars)

        # Gradient clipping
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return losses

    def train_epoch(
        self,
        dataset: tf.data.Dataset,
        epoch: int,
        total_epochs: int,
        steps_per_epoch: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        :param dataset: Training dataset
        :type dataset: tf.data.Dataset
        :param epoch: Current epoch number
        :type epoch: int
        :param total_epochs: Total number of epochs
        :type total_epochs: int
        :param steps_per_epoch: Maximum steps per epoch
        :type steps_per_epoch: Optional[int]
        :return: Dictionary of epoch metrics
        :rtype: Dict[str, float]
        """
        # Reset metrics
        self.train_loss.reset_state()
        self.pred_loss.reset_state()
        self.var_loss.reset_state()
        self.cov_loss.reset_state()

        total_steps = steps_per_epoch or len(dataset)

        for step, images in enumerate(dataset):
            if steps_per_epoch and step >= steps_per_epoch:
                break

            batch_size = tf.shape(images)[0].numpy()

            # Generate masks for this batch
            ctx_mask, tgt_mask, ctx_pos, tgt_pos = generate_batch_masks(
                batch_size=batch_size,
                grid_size=self.grid_size,
                num_targets=self.num_targets,
                target_scale=self.target_scale,
            )

            # Training step (only positions needed now)
            losses = self.train_step(
                images=images,
                ctx_pos=ctx_pos,
                tgt_pos=tgt_pos,
            )

            # Update metrics
            self.train_loss.update_state(losses["total_loss"])
            self.pred_loss.update_state(losses["prediction_loss"])
            if "variance_loss" in losses:
                self.var_loss.update_state(losses["variance_loss"])
            if "covariance_loss" in losses:
                self.cov_loss.update_state(losses["covariance_loss"])

            # Update target encoder EMA
            global_step = epoch * total_steps + step
            total_global_steps = total_epochs * total_steps
            tau = cosine_ema_schedule(
                step=global_step,
                total_steps=total_global_steps,
                base_tau=self.model.ema_decay_base,
                final_tau=self.model.ema_decay_final
            )
            self.model.update_target_encoder(tau)

            # Logging
            if step % 50 == 0:
                print(
                    f"  Step {step}/{total_steps} - "
                    f"Loss: {self.train_loss.result():.4f}, "
                    f"Pred: {self.pred_loss.result():.4f}, "
                    f"Var: {self.var_loss.result():.4f}, "
                    f"Cov: {self.cov_loss.result():.4f}, "
                    f"EMA tau: {tau:.6f}"
                )

        return {
            "total_loss": float(self.train_loss.result()),
            "prediction_loss": float(self.pred_loss.result()),
            "variance_loss": float(self.var_loss.result()),
            "covariance_loss": float(self.cov_loss.result()),
        }

    def fit(
        self,
        train_dataset: tf.data.Dataset,
        epochs: int,
        steps_per_epoch: Optional[int] = None,
        checkpoint_dir: Optional[str] = None,
        checkpoint_freq: int = 10,
    ) -> Dict[str, list]:
        """
        Full training loop.

        :param train_dataset: Training dataset
        :type train_dataset: tf.data.Dataset
        :param epochs: Number of training epochs
        :type epochs: int
        :param steps_per_epoch: Steps per epoch (default: full dataset)
        :type steps_per_epoch: Optional[int]
        :param checkpoint_dir: Directory for saving checkpoints
        :type checkpoint_dir: Optional[str]
        :param checkpoint_freq: Checkpoint frequency (epochs)
        :type checkpoint_freq: int
        :return: Training history
        :rtype: Dict[str, list]
        """
        history = {
            "total_loss": [],
            "prediction_loss": [],
            "variance_loss": [],
            "covariance_loss": [],
        }

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)

            metrics = self.train_epoch(
                dataset=train_dataset,
                epoch=epoch,
                total_epochs=epochs,
                steps_per_epoch=steps_per_epoch,
            )

            # Record history
            for key, value in metrics.items():
                history[key].append(value)

            print(
                f"Epoch {epoch + 1} complete - "
                f"Loss: {metrics['total_loss']:.4f}, "
                f"Pred: {metrics['prediction_loss']:.4f}"
            )

            # Checkpointing
            if checkpoint_dir and (epoch + 1) % checkpoint_freq == 0:
                ckpt_path = f"{checkpoint_dir}/matryoshka_jepa_epoch_{epoch + 1}.keras"
                self.model.save(ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")

        return history


# =============================================================================
# Learning Rate Schedule
# =============================================================================


def create_cosine_lr_schedule(
    base_lr: float,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 1e-6
) -> keras.optimizers.schedules.LearningRateSchedule:
    """
    Create cosine decay learning rate schedule with warmup.

    :param base_lr: Base learning rate
    :type base_lr: float
    :param warmup_steps: Number of warmup steps
    :type warmup_steps: int
    :param total_steps: Total training steps
    :type total_steps: int
    :param min_lr: Minimum learning rate
    :type min_lr: float
    :return: Learning rate schedule
    :rtype: keras.optimizers.schedules.LearningRateSchedule
    """
    warmup_schedule = keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=0.0,
        decay_steps=warmup_steps,
        end_learning_rate=base_lr,
        power=1.0
    )

    cosine_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=base_lr,
        decay_steps=total_steps - warmup_steps,
        alpha=min_lr / base_lr
    )

    class WarmupCosineSchedule(keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, warmup, cosine, warmup_steps):
            super().__init__()
            self.warmup = warmup
            self.cosine = cosine
            self.warmup_steps = warmup_steps

        def __call__(self, step):
            return tf.cond(
                step < self.warmup_steps,
                lambda: self.warmup(step),
                lambda: self.cosine(step - self.warmup_steps)
            )

        def get_config(self):
            return {
                "warmup_steps": self.warmup_steps
            }

    return WarmupCosineSchedule(warmup_schedule, cosine_schedule, warmup_steps)


# =============================================================================
# Main Training Script
# =============================================================================


def train_matryoshka_jepa(
    # Model config
    embed_dim: int = 384,
    encoder_layers: int = 6,
    predictor_layers: int = 4,
    patch_size: int = 16,
    matryoshka_dims: Optional[list] = None,
    # Training config
    batch_size: int = 32,
    epochs: int = 100,
    base_lr: float = 1e-3,
    weight_decay: float = 0.05,
    warmup_epochs: int = 10,
    # Data config
    image_size: int = 224,
    use_cifar: bool = True,
    # Masking config
    num_targets: int = 4,
    target_scale: Tuple[float, float] = (0.15, 0.2),
    # VICReg config
    use_vicreg: bool = True,
    lambda_var: float = 10.0,
    lambda_cov: float = 10.0,
    # Output config
    checkpoint_dir: Optional[str] = None,
) -> Tuple[MatryoshkaJEPA, Dict[str, list]]:
    """
    Train a Matryoshka-JEPA model.

    :param embed_dim: Embedding dimension
    :type embed_dim: int
    :param encoder_layers: Number of encoder layers
    :type encoder_layers: int
    :param predictor_layers: Number of predictor layers
    :type predictor_layers: int
    :param patch_size: Image patch size
    :type patch_size: int
    :param matryoshka_dims: Matryoshka embedding dimensions
    :type matryoshka_dims: Optional[list]
    :param batch_size: Training batch size
    :type batch_size: int
    :param epochs: Number of training epochs
    :type epochs: int
    :param base_lr: Base learning rate
    :type base_lr: float
    :param weight_decay: Weight decay for AdamW
    :type weight_decay: float
    :param warmup_epochs: Number of warmup epochs
    :type warmup_epochs: int
    :param image_size: Input image size
    :type image_size: int
    :param use_cifar: Whether to use CIFAR-10 data
    :type use_cifar: bool
    :param num_targets: Number of target blocks
    :type num_targets: int
    :param target_scale: Target block scale range
    :type target_scale: Tuple[float, float]
    :param use_vicreg: Whether to use VICReg
    :type use_vicreg: bool
    :param lambda_var: VICReg variance weight
    :type lambda_var: float
    :param lambda_cov: VICReg covariance weight
    :type lambda_cov: float
    :param checkpoint_dir: Checkpoint directory
    :type checkpoint_dir: Optional[str]
    :return: Trained model and history
    :rtype: Tuple[MatryoshkaJEPA, Dict[str, list]]
    """
    print("=" * 60)
    print("Matryoshka-JEPA Training")
    print("=" * 60)

    # Default Matryoshka dimensions
    if matryoshka_dims is None:
        matryoshka_dims = [64, 128, 256, embed_dim] if embed_dim >= 256 else [embed_dim]

    print(f"\nModel Configuration:")
    print(f"  - Embed dim: {embed_dim}")
    print(f"  - Encoder layers: {encoder_layers}")
    print(f"  - Predictor layers: {predictor_layers}")
    print(f"  - Patch size: {patch_size}")
    print(f"  - Matryoshka dims: {matryoshka_dims}")
    print(f"  - VICReg: {use_vicreg} (var={lambda_var}, cov={lambda_cov})")

    # Create model
    model = MatryoshkaJEPA(
        embed_dim=embed_dim,
        encoder_layers=encoder_layers,
        predictor_layers=predictor_layers,
        patch_size=patch_size,
        matryoshka_dims=matryoshka_dims,
        use_vicreg=use_vicreg,
        lambda_var=lambda_var,
        lambda_cov=lambda_cov,
    )

    # Build model with dummy input
    dummy_input = np.zeros((1, image_size, image_size, 3), dtype=np.float32)
    _ = model(dummy_input)

    print(f"\nModel built successfully")
    print(f"  - Context encoder params: {model.context_encoder.count_params():,}")
    print(f"  - Predictor params: {model.predictor.count_params():,}")
    print(f"  - Total params: {model.count_params():,}")

    # Create dataset
    if use_cifar:
        train_ds, val_ds = create_cifar10_dataset(
            batch_size=batch_size,
            image_size=image_size
        )
        steps_per_epoch = 50000 // batch_size
    else:
        train_ds = create_random_image_dataset(
            batch_size=batch_size,
            image_size=image_size,
            num_samples=1000
        )
        steps_per_epoch = 1000 // batch_size

    # Learning rate schedule
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    lr_schedule = create_cosine_lr_schedule(
        base_lr=base_lr,
        warmup_steps=warmup_steps,
        total_steps=total_steps
    )

    # Optimizer
    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=weight_decay,
        beta_1=0.9,
        beta_2=0.95,
    )

    print(f"\nTraining Configuration:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Steps/epoch: {steps_per_epoch}")
    print(f"  - Total steps: {total_steps}")
    print(f"  - Warmup steps: {warmup_steps}")
    print(f"  - Base LR: {base_lr}")
    print(f"  - Weight decay: {weight_decay}")

    # Create trainer
    grid_size = image_size // patch_size
    trainer = MatryoshkaJEPATrainer(
        model=model,
        optimizer=optimizer,
        grid_size=grid_size,
        num_targets=num_targets,
        target_scale=target_scale,
    )

    # Train
    print("\nStarting training...")
    history = trainer.fit(
        train_dataset=train_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        checkpoint_dir=checkpoint_dir,
    )

    print("\nTraining complete!")

    return model, history


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    # Small model for quick testing
    model, history = train_matryoshka_jepa(
        # Smaller model for testing
        embed_dim=192,
        encoder_layers=4,
        predictor_layers=2,
        patch_size=16,
        matryoshka_dims=[48, 96, 192],
        # Quick training
        batch_size=16,
        epochs=5,
        base_lr=5e-4,
        warmup_epochs=1,
        # Data
        image_size=224,
        use_cifar=True,
        # Masking
        num_targets=4,
        target_scale=(0.15, 0.2),
        # VICReg
        use_vicreg=True,
        lambda_var=5.0,
        lambda_cov=5.0,
    )

    # Test Matryoshka inference at different dimensions
    print("\n" + "=" * 60)
    print("Testing Matryoshka Inference")
    print("=" * 60)

    test_image = np.random.randn(1, 224, 224, 3).astype(np.float32)

    for dim in [48, 96, 192]:
        features = model.get_features(test_image, dim=dim)
        print(f"Dim {dim}: feature shape = {features.shape}")

    # Save final model
    model.save("/tmp/matryoshka_jepa_final.keras")
    print("\nModel saved to /tmp/matryoshka_jepa_final.keras")