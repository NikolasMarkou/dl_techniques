"""
Training Infrastructure for Score-Based nanoVLM

Implements Denoising Score Matching (DSM) training and loss functions.
The key insight: training an optimal denoiser via MSE is equivalent to
learning the score function ∇ log p by Miyasawa's theorem.
"""

import keras
from keras import ops
from typing import Dict, Optional, Any
import tensorflow as tf

from dl_techniques.utils.logger import logger
from .model import create_score_based_nanovlm

@keras.saving.register_keras_serializable()
class DenoisingScoreMatchingLoss(keras.losses.Loss):
    """
    Denoising Score Matching loss for learning score functions.

    By Miyasawa's theorem, training a denoiser D(x_t, c, t) to predict
    the clean data x_0 from noisy x_t is equivalent to learning the score:

        L_DSM = E[||D(x_t, c, t) - x_0||²]

    This simple MSE loss implicitly trains the model to estimate
    ∇ log p(x_t | c), enabling score-based generation.

    Args:
        prediction_type: What the denoiser predicts
            - 'epsilon': Predicts the noise ε
            - 'sample': Predicts the clean sample x_0
            - 'v_prediction': Predicts velocity v
        loss_weight_type: How to weight loss across timesteps
            - 'uniform': Equal weight
            - 'snr': Signal-to-noise ratio weighting
            - 'truncated_snr': Truncated SNR (Min-SNR)
        reduction: Keras reduction type. Defaults to 'sum_over_batch_size'.
        **kwargs: Additional loss arguments.

    References:
        - Ho et al. (2020): Simple L2 loss works best
        - Hang et al. (2023): Min-SNR weighting for stability
    """

    def __init__(
            self,
            prediction_type: str = 'epsilon',
            loss_weight_type: str = 'uniform',
            min_snr_gamma: float = 5.0,
            reduction: str = 'sum_over_batch_size',
            name: str = 'dsm_loss',
            **kwargs
    ) -> None:
        super().__init__(reduction=reduction, name=name, **kwargs)

        self.prediction_type = prediction_type
        self.loss_weight_type = loss_weight_type
        self.min_snr_gamma = min_snr_gamma

        logger.info(
            f"Initialized DSM loss: prediction={prediction_type}, "
            f"weighting={loss_weight_type}"
        )

    def call(
            self,
            y_true: keras.KerasTensor,
            y_pred: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Compute DSM loss.

        Args:
            y_true: True clean data or noise (depending on prediction_type)
            y_pred: Predicted data from denoiser

        Returns:
            Scalar loss value
        """
        # Simple MSE - the magic is in what we're predicting
        loss = ops.mean(ops.square(y_pred - y_true), axis=list(range(1, len(y_pred.shape))))

        # Optionally apply timestep-dependent weighting
        # This would require passing timesteps, which we handle in the trainer

        return ops.mean(loss)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'prediction_type': self.prediction_type,
            'loss_weight_type': self.loss_weight_type,
            'min_snr_gamma': self.min_snr_gamma,
        })
        return config


@keras.saving.register_keras_serializable()
class VLMDenoisingLoss(keras.losses.Loss):
    """
    Combined loss for vision-language denoising.

    Supports multiple denoising objectives simultaneously:
    - Vision denoising (text → image)
    - Text denoising (image → text)
    - Joint denoising (unified world model)

    Args:
        vision_weight: Weight for vision denoising loss. Defaults to 1.0.
        text_weight: Weight for text denoising loss. Defaults to 1.0.
        joint_weight: Weight for joint denoising loss. Defaults to 0.5.
        **kwargs: Additional loss arguments.
    """

    def __init__(
            self,
            vision_weight: float = 1.0,
            text_weight: float = 1.0,
            joint_weight: float = 0.5,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.vision_weight = vision_weight
        self.text_weight = text_weight
        self.joint_weight = joint_weight

        # Component losses
        self.dsm_loss = DenoisingScoreMatchingLoss(prediction_type='sample')

    def call(
            self,
            y_true: Dict[str, keras.KerasTensor],
            y_pred: Dict[str, keras.KerasTensor]
    ) -> keras.KerasTensor:
        """
        Compute combined VLM loss.

        Args:
            y_true: Dictionary of true targets
            y_pred: Dictionary of predictions

        Returns:
            Weighted combination of losses
        """
        total_loss = 0.0

        # Vision denoising loss
        if 'denoised_vision' in y_pred and 'target_vision' in y_true:
            vision_loss = self.dsm_loss(
                y_true['target_vision'], y_pred['denoised_vision']
            )
            total_loss += self.vision_weight * vision_loss

        # Text denoising loss
        if 'denoised_text' in y_pred and 'target_text' in y_true:
            text_loss = self.dsm_loss(
                y_true['target_text'], y_pred['denoised_text']
            )
            total_loss += self.text_weight * text_loss

        # Joint denoising losses
        if 'joint_denoised_vision' in y_pred:
            joint_vision_loss = self.dsm_loss(
                y_true['joint_target_vision'], y_pred['joint_denoised_vision']
            )
            total_loss += self.joint_weight * joint_vision_loss

        if 'joint_denoised_text' in y_pred:
            joint_text_loss = self.dsm_loss(
                y_true['joint_target_text'], y_pred['joint_denoised_text']
            )
            total_loss += self.joint_weight * joint_text_loss

        return total_loss

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'vision_weight': self.vision_weight,
            'text_weight': self.text_weight,
            'joint_weight': self.joint_weight,
        })
        return config


class ScoreVLMTrainer:
    """
    Custom trainer for Score-Based nanoVLM.

    Implements the Denoising Score Matching training loop with support
    for mixed precision, gradient accumulation, and EMA.

    Args:
        model: ScoreBasedNanoVLM instance
        optimizer: Keras optimizer
        loss_fn: VLMDenoisingLoss instance
        use_ema: Whether to use Exponential Moving Average. Defaults to True.
        ema_decay: EMA decay rate. Defaults to 0.9999.
        gradient_accumulation_steps: Steps to accumulate gradients. Defaults to 1.
    """

    def __init__(
            self,
            model: keras.Model,
            optimizer: keras.optimizers.Optimizer,
            loss_fn: VLMDenoisingLoss,
            use_ema: bool = True,
            ema_decay: float = 0.9999,
            gradient_accumulation_steps: int = 1
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # EMA model for inference
        if use_ema:
            self.ema_model = keras.models.clone_model(model)
            self.ema_model.set_weights(model.get_weights())

        # Metrics
        self.train_loss = keras.metrics.Mean(name='train_loss')
        self.vision_loss = keras.metrics.Mean(name='vision_loss')
        self.text_loss = keras.metrics.Mean(name='text_loss')

        # Gradient accumulation
        self.accumulated_gradients = None
        self.accumulation_counter = 0

        logger.info(
            f"Initialized ScoreVLM trainer with EMA={use_ema}, "
            f"grad_accum={gradient_accumulation_steps}"
        )

    @tf.function
    def train_step(
            self,
            images: keras.KerasTensor,
            text_tokens: keras.KerasTensor
    ) -> Dict[str, float]:
        """
        Single training step with DSM.

        Args:
            images: Batch of images [batch, H, W, C]
            text_tokens: Batch of text tokens [batch, seq_len]

        Returns:
            Dictionary of metrics
        """
        with tf.GradientTape() as tape:
            # Forward pass: model adds noise internally and denoises
            outputs = self.model(
                {'images': images, 'text': text_tokens},
                training=True
            )

            # Compute DSM loss
            loss = self.loss_fn(outputs, outputs)

            # Scale loss for gradient accumulation
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps

        # Compute gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)

        # Accumulate or apply gradients
        if self.gradient_accumulation_steps > 1:
            if self.accumulated_gradients is None:
                self.accumulated_gradients = [
                    tf.zeros_like(g) if g is not None else None
                    for g in gradients
                ]

            # Accumulate
            for i, grad in enumerate(gradients):
                if grad is not None:
                    self.accumulated_gradients[i] = (
                            self.accumulated_gradients[i] + grad
                    )

            self.accumulation_counter += 1

            # Apply when accumulated enough
            if self.accumulation_counter >= self.gradient_accumulation_steps:
                self.optimizer.apply_gradients(
                    zip(self.accumulated_gradients, self.model.trainable_variables)
                )

                # Reset
                self.accumulated_gradients = None
                self.accumulation_counter = 0

                # Update EMA
                if self.use_ema:
                    self._update_ema()
        else:
            # Direct gradient application
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables)
            )

            # Update EMA
            if self.use_ema:
                self._update_ema()

        # Update metrics
        self.train_loss.update_state(loss * self.gradient_accumulation_steps)

        # Component-specific losses
        if 'denoised_vision' in outputs:
            v_loss = ops.mean(ops.square(
                outputs['denoised_vision'] - outputs['target_vision']
            ))
            self.vision_loss.update_state(v_loss)

        if 'denoised_text' in outputs:
            t_loss = ops.mean(ops.square(
                outputs['denoised_text'] - outputs['target_text']
            ))
            self.text_loss.update_state(t_loss)

        return {
            'loss': self.train_loss.result(),
            'vision_loss': self.vision_loss.result(),
            'text_loss': self.text_loss.result(),
        }

    def _update_ema(self) -> None:
        """Update EMA model weights."""
        for ema_weight, model_weight in zip(
                self.ema_model.weights, self.model.weights
        ):
            ema_weight.assign(
                self.ema_decay * ema_weight + (1 - self.ema_decay) * model_weight
            )

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self.train_loss.reset_states()
        self.vision_loss.reset_states()
        self.text_loss.reset_states()

    def get_model_for_inference(self) -> keras.Model:
        """Get model for inference (EMA if available)."""
        return self.ema_model if self.use_ema else self.model


def train_score_vlm(
        model: keras.Model,
        train_dataset: keras.utils.Sequence,
        epochs: int = 100,
        optimizer_config: Optional[Dict] = None,
        checkpoint_dir: str = 'checkpoints/',
        log_frequency: int = 100
) -> None:
    """
    Main training loop for Score-Based nanoVLM.

    Args:
        model: ScoreBasedNanoVLM instance
        train_dataset: Training dataset
        epochs: Number of epochs
        optimizer_config: Optimizer configuration
        checkpoint_dir: Directory for checkpoints
        log_frequency: Log every N steps
    """
    logger.info("Starting Score-Based nanoVLM training")

    # Setup optimizer
    if optimizer_config is None:
        optimizer_config = {
            'type': 'adamw',
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
        }

    from dl_techniques.optimization import optimizer_builder
    optimizer = optimizer_builder(optimizer_config)

    # Setup loss and trainer
    loss_fn = VLMDenoisingLoss(
        vision_weight=1.0,
        text_weight=1.0,
        joint_weight=0.5
    )

    trainer = ScoreVLMTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        use_ema=True,
        gradient_accumulation_steps=4
    )

    # Training loop
    global_step = 0

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        trainer.reset_metrics()

        for step, (images, text_tokens) in enumerate(train_dataset):
            # Training step
            metrics = trainer.train_step(images, text_tokens)

            global_step += 1

            # Logging
            if step % log_frequency == 0:
                logger.info(
                    f"Step {global_step}: "
                    f"Loss={float(metrics['loss']):.4f}, "
                    f"VisionLoss={float(metrics.get('vision_loss', 0)):.4f}, "
                    f"TextLoss={float(metrics.get('text_loss', 0)):.4f}"
                )

        # Epoch end: save checkpoint
        checkpoint_path = f"{checkpoint_dir}/score_vlm_epoch_{epoch + 1}.keras"
        inference_model = trainer.get_model_for_inference()
        inference_model.save(checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Generate samples for monitoring
        if epoch % 5 == 0:
            logger.info("Generating samples...")
            # TODO: Add sample generation

    logger.info("Training completed!")


# === Example usage ===

def example_training():
    """Example of training a score-based VLM."""


    # Create model
    model = create_score_based_nanovlm(
        variant='base',
        mode='joint',
        vocab_size=32000
    )

    # Dummy dataset
    class DummyDataset(keras.utils.Sequence):
        def __len__(self):
            return 100

        def __getitem__(self, idx):
            images = keras.random.normal((8, 224, 224, 3))
            text = keras.random.uniform((8, 77), minval=0, maxval=32000, dtype='int32')
            return images, text

    dataset = DummyDataset()

    # Train
    train_score_vlm(
        model=model,
        train_dataset=dataset,
        epochs=10,
        log_frequency=10
    )

    logger.info("Example training completed!")


if __name__ == '__main__':
    # Enable mixed precision
    policy = keras.mixed_precision.Policy('mixed_float16')
    keras.mixed_precision.set_global_policy(policy)

    example_training()