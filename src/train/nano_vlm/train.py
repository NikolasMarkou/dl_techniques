"""Training script for nanoVLM model with multi-optimizer support."""

import keras
from keras import ops
from typing import Dict, List, Tuple, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.losses.nano_vlm_loss import NanoVLMLoss
from dl_techniques.models.nano_vlm.model import create_nanovlm
from dl_techniques.utils.datasets.vqa_dataset import VQADataProcessor, load_cauldron_sample
from dl_techniques.optimization import optimizer_builder, learning_rate_schedule_builder


# ---------------------------------------------------------------------

def create_training_setup() -> Dict[str, Any]:
    """Create training configuration for nanoVLM.

    Returns:
        Dict[str, Any]: Dictionary containing optimizer, learning rate schedule, and loss function.
    """
    # Learning rate schedule configuration
    lr_config = {
        "type": "cosine_decay",
        "warmup_steps": 1000,
        "warmup_start_lr": 1e-8,
        "learning_rate": 1e-4,  # Base learning rate
        "decay_steps": 50000,
        "alpha": 0.0001
    }

    # Optimizer configuration
    optimizer_config = {
        "type": "adamw",
        "beta_1": 0.9,
        "beta_2": 0.999,
        "gradient_clipping_by_norm": 1.0
    }

    # Build components
    lr_schedule = learning_rate_schedule_builder(lr_config)
    optimizer = optimizer_builder(optimizer_config, lr_schedule)

    return {
        "optimizer": optimizer,
        "lr_schedule": lr_schedule,
        "loss_fn": NanoVLMLoss(ignore_index=0, label_smoothing=0.1)
    }

# ---------------------------------------------------------------------

def setup_different_learning_rates(model: keras.Model) -> Dict[str, Any]:
    """Setup different learning rates for different model components.

    Args:
        model (keras.Model): The nanoVLM model to configure optimizers for.

    Returns:
        Dict[str, Any]: Dictionary containing optimizers and parameter groups.
    """
    # Separate parameters by component
    vision_params: List[keras.Variable] = []
    language_params: List[keras.Variable] = []
    projection_params: List[keras.Variable] = []

    for layer in model.layers:
        layer_name = layer.name.lower()
        if 'vision_heads' in layer_name or 'encoder' in layer_name:
            vision_params.extend(layer.trainable_variables)
        elif 'projection' in layer_name or 'connector' in layer_name:
            projection_params.extend(layer.trainable_variables)
        else:
            language_params.extend(layer.trainable_variables)

    # Create optimizers with different learning rates
    vision_lr_config = {
        "type": "cosine_decay",
        "warmup_steps": 500,
        "warmup_start_lr": 1e-9,
        "learning_rate": 1e-5,  # Lower for pre-trained vision_heads
        "decay_steps": 25000,
        "alpha": 0.0001
    }

    language_lr_config = {
        "type": "cosine_decay",
        "warmup_steps": 500,
        "warmup_start_lr": 1e-9,
        "learning_rate": 1e-5,  # Lower for pre-trained language
        "decay_steps": 25000,
        "alpha": 0.0001
    }

    projection_lr_config = {
        "type": "cosine_decay",
        "warmup_steps": 1000,
        "warmup_start_lr": 1e-8,
        "learning_rate": 1e-4,  # Higher for new projection layers
        "decay_steps": 50000,
        "alpha": 0.0001
    }

    # Build optimizers
    vision_lr_schedule = learning_rate_schedule_builder(vision_lr_config)
    language_lr_schedule = learning_rate_schedule_builder(language_lr_config)
    projection_lr_schedule = learning_rate_schedule_builder(projection_lr_config)

    vision_optimizer_config = {
        "type": "adamw",
        "beta_1": 0.9,
        "beta_2": 0.999,
        "gradient_clipping_by_norm": 0.5
    }

    language_optimizer_config = {
        "type": "adamw",
        "beta_1": 0.9,
        "beta_2": 0.999,
        "gradient_clipping_by_norm": 0.5
    }

    projection_optimizer_config = {
        "type": "adamw",
        "beta_1": 0.9,
        "beta_2": 0.999,
        "gradient_clipping_by_norm": 1.0
    }

    vision_optimizer = optimizer_builder(vision_optimizer_config, vision_lr_schedule)
    language_optimizer = optimizer_builder(language_optimizer_config, language_lr_schedule)
    projection_optimizer = optimizer_builder(projection_optimizer_config, projection_lr_schedule)

    return {
        "vision_optimizer": vision_optimizer,
        "language_optimizer": language_optimizer,
        "projection_optimizer": projection_optimizer,
        "vision_params": vision_params,
        "language_params": language_params,
        "projection_params": projection_params
    }

# ---------------------------------------------------------------------

class NanoVLMTrainer:
    """Custom trainer for nanoVLM with multi-optimizer support.

    This trainer implements differential learning rates for different components
    of the nanoVLM model (vision_heads encoder, language model, projection layers).

    Args:
        model (keras.Model): The nanoVLM model to train.
        loss_fn (keras.losses.Loss): Loss function for training.
        use_multi_optimizer (bool, optional): Whether to use multiple optimizers
            with different learning rates. Defaults to True.
    """

    def __init__(
            self,
            model: keras.Model,
            loss_fn: keras.losses.Loss,
            use_multi_optimizer: bool = True
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.use_multi_optimizer = use_multi_optimizer

        # Setup optimizers
        if use_multi_optimizer:
            self.optimizers = setup_different_learning_rates(model)
        else:
            training_setup = create_training_setup()
            self.single_optimizer = training_setup["optimizer"]

        # Initialize metrics
        self.train_loss = keras.metrics.Mean(name='train_loss')
        self.train_accuracy = keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    def _apply_multi_optimizer_gradients(
            self,
            gradients: List[keras.Variable],
            variables: List[keras.Variable]
    ) -> None:
        """Apply gradients using multiple optimizers for different components.

        Args:
            gradients (List[keras.Variable]): Computed gradients.
            variables (List[keras.Variable]): Model variables.
        """
        # Split gradients by component
        vision_grads = []
        language_grads = []
        projection_grads = []

        vision_vars = []
        language_vars = []
        projection_vars = []

        for grad, var in zip(gradients, variables):
            if grad is None:  # Skip None gradients
                continue

            if var in self.optimizers['vision_params']:
                vision_grads.append(grad)
                vision_vars.append(var)
            elif var in self.optimizers['projection_params']:
                projection_grads.append(grad)
                projection_vars.append(var)
            elif var in self.optimizers['language_params']:
                language_grads.append(grad)
                language_vars.append(var)

        # Apply gradients with respective optimizers
        if vision_grads and vision_vars:
            self.optimizers['vision_optimizer'].apply_gradients(
                zip(vision_grads, vision_vars)
            )
        if language_grads and language_vars:
            self.optimizers['language_optimizer'].apply_gradients(
                zip(language_grads, language_vars)
            )
        if projection_grads and projection_vars:
            self.optimizers['projection_optimizer'].apply_gradients(
                zip(projection_grads, projection_vars)
            )

    def train_step(self, batch_data: Tuple[Any, Any]) -> Dict[str, keras.Variable]:
        """Custom training step with support for multiple optimizers.

        Args:
            batch_data (Tuple[Any, Any]): Batch containing inputs and labels.

        Returns:
            Dict[str, keras.Variable]: Dictionary containing loss and accuracy metrics.
        """
        inputs, labels = batch_data

        with keras.GradientTape() as tape:
            predictions = self.model(inputs, training=True)
            loss = self.loss_fn(labels, predictions)

            # Apply loss scaling for mixed precision if enabled
            if isinstance(self.model.dtype_policy, keras.mixed_precision.Policy):
                if self.model.dtype_policy.name == 'mixed_float16':
                    loss = ops.cast(loss, dtype='float32')

        # Compute gradients
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Apply gradients
        if self.use_multi_optimizer:
            self._apply_multi_optimizer_gradients(gradients, trainable_vars)
        else:
            # Filter out None gradients
            filtered_grads_and_vars = [
                (grad, var) for grad, var in zip(gradients, trainable_vars)
                if grad is not None
            ]
            if filtered_grads_and_vars:
                self.single_optimizer.apply_gradients(filtered_grads_and_vars)

        # Update metrics
        self.train_loss.update_state(loss)
        self.train_accuracy.update_state(labels, predictions)

        return {
            'loss': self.train_loss.result(),
            'accuracy': self.train_accuracy.result()
        }

    def reset_metrics(self) -> None:
        """Reset training metrics."""
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()

# ---------------------------------------------------------------------

def configure_mixed_precision() -> None:
    """Configure mixed precision training for better performance."""
    try:
        # Check if mixed precision is supported
        policy = keras.mixed_precision.Policy('mixed_float16')
        keras.mixed_precision.set_global_policy(policy)
        logger.info("Mixed precision training enabled (float16)")
    except Exception as e:
        logger.warning(f"Mixed precision not available: {e}. Using float32.")
        # Fallback to float32
        policy = keras.mixed_precision.Policy('float32')
        keras.mixed_precision.set_global_policy(policy)

# ---------------------------------------------------------------------

def train_nanovlm(
        epochs: int = 10,
        batch_size: int = 8,
        use_multi_optimizer: bool = True,
        checkpoint_frequency: int = 5,
        log_frequency: int = 10
) -> None:
    """Main training function for nanoVLM.

    Args:
        epochs (int, optional): Number of training epochs. Defaults to 10.
        batch_size (int, optional): Training batch size. Defaults to 8.
        use_multi_optimizer (bool, optional): Whether to use multiple optimizers. Defaults to True.
        checkpoint_frequency (int, optional): Save checkpoint every N epochs. Defaults to 5.
        log_frequency (int, optional): Log metrics every N steps. Defaults to 10.
    """
    logger.info("Starting nanoVLM training")

    # Configure mixed precision
    configure_mixed_precision()

    # Create model
    try:
        model = create_nanovlm()
        logger.info("Created nanoVLM-222M model")
        logger.info(f"Model has {model.count_params():,} parameters")
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        raise

    # Setup training
    training_setup = create_training_setup()
    trainer = NanoVLMTrainer(
        model,
        training_setup['loss_fn'],
        use_multi_optimizer=use_multi_optimizer
    )

    logger.info(f"Using {'multi-optimizer' if use_multi_optimizer else 'single optimizer'} training")

    # Prepare data
    try:
        data_processor = VQADataProcessor(
            image_size=224,
            max_text_length=512,
            vocab_size=32000
        )

        # Load sample data (replace with real dataset)
        sample_data = load_cauldron_sample()
        train_dataset = data_processor.create_tensorflow_dataset(
            sample_data,
            batch_size=batch_size,
            shuffle=True
        )

        steps_per_epoch = len(train_dataset)
        logger.info(f"Dataset prepared: {steps_per_epoch} steps per epoch")

    except Exception as e:
        logger.error(f"Failed to prepare dataset: {e}")
        raise

    logger.info(
        f"Training configuration:\n"
        f"  Epochs: {epochs}\n"
        f"  Batch size: {batch_size}\n"
        f"  Steps per epoch: {steps_per_epoch}\n"
        f"  Total training steps: {epochs * steps_per_epoch}"
    )

    # Training loop
    try:
        for epoch in range(epochs):
            logger.info(f"Starting epoch {epoch + 1}/{epochs}")

            # Reset metrics
            trainer.reset_metrics()

            # Train for one epoch
            for step, batch in enumerate(train_dataset):
                try:
                    metrics = trainer.train_step(batch)

                    if step % log_frequency == 0:
                        logger.info(
                            f"Epoch {epoch + 1}/{epochs}, Step {step}/{steps_per_epoch}: "
                            f"Loss = {float(metrics['loss']):.4f}, "
                            f"Accuracy = {float(metrics['accuracy']):.4f}"
                        )

                except Exception as e:
                    logger.error(f"Error in training step {step}: {e}")
                    continue

            # End of epoch logging
            final_loss = float(trainer.train_loss.result())
            final_accuracy = float(trainer.train_accuracy.result())

            logger.info(
                f"Epoch {epoch + 1} completed: "
                f"Loss = {final_loss:.4f}, "
                f"Accuracy = {final_accuracy:.4f}"
            )

            # Save checkpoint
            if checkpoint_frequency > 0 and (epoch + 1) % checkpoint_frequency == 0:
                try:
                    checkpoint_path = f"nanovlm_checkpoint_epoch_{epoch + 1}.keras"
                    model.save(checkpoint_path)
                    logger.info(f"Saved checkpoint: {checkpoint_path}")
                except Exception as e:
                    logger.error(f"Failed to save checkpoint: {e}")

        # Save final model
        try:
            final_model_path = "nanovlm_final.keras"
            model.save(final_model_path)
            logger.info(f"Training completed. Final model saved: {final_model_path}")
        except Exception as e:
            logger.error(f"Failed to save final model: {e}")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        # Save emergency checkpoint
        try:
            emergency_path = "nanovlm_emergency_checkpoint.keras"
            model.save(emergency_path)
            logger.info(f"Emergency checkpoint saved: {emergency_path}")
        except Exception as e:
            logger.error(f"Failed to save emergency checkpoint: {e}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Configure training parameters
    training_config = {
        "epochs": 10,
        "batch_size": 8,
        "use_multi_optimizer": True,
        "checkpoint_frequency": 5,
        "log_frequency": 10
    }

    train_nanovlm(**training_config)

# ---------------------------------------------------------------------
