"""Training script for nanoVLM model with multi-optimizer support."""

import os
import argparse
import keras
from keras import ops
from datetime import datetime
from typing import Dict, List, Tuple, Any

from train.common import setup_gpu
from dl_techniques.utils.logger import logger
from dl_techniques.losses.nano_vlm_loss import NanoVLMLoss
from dl_techniques.models.nano_vlm.model import create_nanovlm
from dl_techniques.datasets.vqa_dataset import VQADataProcessor, load_cauldron_sample, create_vqa_dataset
from dl_techniques.optimization import optimizer_builder, learning_rate_schedule_builder


# ---------------------------------------------------------------------

def create_training_setup() -> Dict[str, Any]:
    """Create training configuration (optimizer, LR schedule, loss)."""
    lr_schedule = learning_rate_schedule_builder({
        "type": "cosine_decay", "warmup_steps": 1000, "warmup_start_lr": 1e-8,
        "learning_rate": 1e-4, "decay_steps": 50000, "alpha": 0.0001
    })

    optimizer = optimizer_builder({
        "type": "adamw", "beta_1": 0.9, "beta_2": 0.999,
        "gradient_clipping_by_norm": 1.0
    }, lr_schedule)

    return {
        "optimizer": optimizer,
        "lr_schedule": lr_schedule,
        "loss_fn": NanoVLMLoss(ignore_index=0, label_smoothing=0.1)
    }


# ---------------------------------------------------------------------

def setup_different_learning_rates(model: keras.Model) -> Dict[str, Any]:
    """Setup different learning rates for vision, language, and projection components."""
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

    def _build_optimizer(lr: float, decay_steps: int, warmup: int, clip: float):
        lr_schedule = learning_rate_schedule_builder({
            "type": "cosine_decay", "warmup_steps": warmup, "warmup_start_lr": 1e-9,
            "learning_rate": lr, "decay_steps": decay_steps, "alpha": 0.0001
        })
        return optimizer_builder({
            "type": "adamw", "beta_1": 0.9, "beta_2": 0.999,
            "gradient_clipping_by_norm": clip
        }, lr_schedule)

    return {
        "vision_optimizer": _build_optimizer(1e-5, 25000, 500, 0.5),
        "language_optimizer": _build_optimizer(1e-5, 25000, 500, 0.5),
        "projection_optimizer": _build_optimizer(1e-4, 50000, 1000, 1.0),
        "vision_params": vision_params,
        "language_params": language_params,
        "projection_params": projection_params
    }


# ---------------------------------------------------------------------

class NanoVLMTrainer:
    """Custom trainer for nanoVLM with multi-optimizer support.

    Implements differential learning rates for vision encoder,
    language model, and projection layers.
    """

    def __init__(self, model: keras.Model, loss_fn: keras.losses.Loss,
                 use_multi_optimizer: bool = True) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.use_multi_optimizer = use_multi_optimizer

        if use_multi_optimizer:
            self.optimizers = setup_different_learning_rates(model)
        else:
            self.single_optimizer = create_training_setup()["optimizer"]

        self.train_loss = keras.metrics.Mean(name='train_loss')
        self.train_accuracy = keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    def _apply_multi_optimizer_gradients(
        self, gradients: List[keras.Variable], variables: List[keras.Variable]
    ) -> None:
        """Apply gradients using component-specific optimizers."""
        component_map = {
            'vision': ([], [], self.optimizers['vision_optimizer'], self.optimizers['vision_params']),
            'language': ([], [], self.optimizers['language_optimizer'], self.optimizers['language_params']),
            'projection': ([], [], self.optimizers['projection_optimizer'], self.optimizers['projection_params']),
        }

        for grad, var in zip(gradients, variables):
            if grad is None:
                continue
            if var in self.optimizers['vision_params']:
                component_map['vision'][0].append(grad)
                component_map['vision'][1].append(var)
            elif var in self.optimizers['projection_params']:
                component_map['projection'][0].append(grad)
                component_map['projection'][1].append(var)
            elif var in self.optimizers['language_params']:
                component_map['language'][0].append(grad)
                component_map['language'][1].append(var)

        for _, (grads, vars, opt, _) in component_map.items():
            if grads and vars:
                opt.apply_gradients(zip(grads, vars))

    def train_step(self, batch_data: Tuple[Any, Any]) -> Dict[str, keras.Variable]:
        inputs, labels = batch_data

        with keras.GradientTape() as tape:
            predictions = self.model(inputs, training=True)
            loss = self.loss_fn(labels, predictions)

            if isinstance(self.model.dtype_policy, keras.mixed_precision.Policy):
                if self.model.dtype_policy.name == 'mixed_float16':
                    loss = ops.cast(loss, dtype='float32')

        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        if self.use_multi_optimizer:
            self._apply_multi_optimizer_gradients(gradients, trainable_vars)
        else:
            filtered = [(g, v) for g, v in zip(gradients, trainable_vars) if g is not None]
            if filtered:
                self.single_optimizer.apply_gradients(filtered)

        self.train_loss.update_state(loss)
        self.train_accuracy.update_state(labels, predictions)

        return {'loss': self.train_loss.result(), 'accuracy': self.train_accuracy.result()}

    def reset_metrics(self) -> None:
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()


# ---------------------------------------------------------------------

def configure_mixed_precision() -> None:
    """Configure mixed precision training."""
    try:
        keras.mixed_precision.set_global_policy(keras.mixed_precision.Policy('mixed_float16'))
        logger.info("Mixed precision enabled (float16)")
    except Exception as e:
        logger.warning(f"Mixed precision not available: {e}. Using float32.")
        keras.mixed_precision.set_global_policy(keras.mixed_precision.Policy('float32'))


# ---------------------------------------------------------------------

def train_nanovlm(
    epochs: int = 10, batch_size: int = 8, use_multi_optimizer: bool = True,
    checkpoint_frequency: int = 5, log_frequency: int = 10,
    gpu_index: int = None,
) -> None:
    """Main training function for nanoVLM."""
    logger.info("Starting nanoVLM training")
    setup_gpu(gpu_id=gpu_index)
    configure_mixed_precision()

    model = create_nanovlm()
    logger.info(f"Created nanoVLM-222M: {model.count_params():,} parameters")

    training_setup = create_training_setup()
    trainer = NanoVLMTrainer(model, training_setup['loss_fn'], use_multi_optimizer=use_multi_optimizer)
    logger.info(f"Using {'multi-optimizer' if use_multi_optimizer else 'single optimizer'} training")

    data_processor = VQADataProcessor(image_size=224, max_text_length=512, vocab_size=32000)
    sample_data = load_cauldron_sample()
    train_dataset = create_vqa_dataset(sample_data, data_processor, batch_size=batch_size, shuffle=True)
    steps_per_epoch = len(train_dataset)

    logger.info(f"Epochs: {epochs}, Batch: {batch_size}, Steps/epoch: {steps_per_epoch}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"nanovlm_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    try:
        for epoch in range(epochs):
            trainer.reset_metrics()

            for step, batch in enumerate(train_dataset):
                try:
                    metrics = trainer.train_step(batch)
                    if step % log_frequency == 0:
                        logger.info(
                            f"Epoch {epoch + 1}/{epochs}, Step {step}/{steps_per_epoch}: "
                            f"Loss={float(metrics['loss']):.4f}, Acc={float(metrics['accuracy']):.4f}"
                        )
                except Exception as e:
                    logger.error(f"Error in step {step}: {e}")
                    continue

            logger.info(
                f"Epoch {epoch + 1} done: Loss={float(trainer.train_loss.result()):.4f}, "
                f"Acc={float(trainer.train_accuracy.result()):.4f}"
            )

            if checkpoint_frequency > 0 and (epoch + 1) % checkpoint_frequency == 0:
                try:
                    path = os.path.join(results_dir, f"checkpoint_epoch_{epoch + 1}.keras")
                    model.save(path)
                    logger.info(f"Saved checkpoint: {path}")
                except Exception as e:
                    logger.error(f"Failed to save checkpoint: {e}")

        final_path = os.path.join(results_dir, "final_model.keras")
        model.save(final_path)
        logger.info(f"Training completed. Final model saved: {final_path}")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        try:
            model.save(os.path.join(results_dir, "emergency_checkpoint.keras"))
            logger.info("Emergency checkpoint saved")
        except Exception as e:
            logger.error(f"Failed to save emergency checkpoint: {e}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train nanoVLM")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--multi-optimizer", action="store_true", default=True)
    parser.add_argument("--no-multi-optimizer", dest="multi_optimizer", action="store_false")
    parser.add_argument("--checkpoint-frequency", type=int, default=5)
    parser.add_argument("--log-frequency", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=None, help="GPU device index")
    args = parser.parse_args()

    train_nanovlm(
        epochs=args.epochs, batch_size=args.batch_size,
        use_multi_optimizer=args.multi_optimizer,
        checkpoint_frequency=args.checkpoint_frequency,
        log_frequency=args.log_frequency,
        gpu_index=args.gpu,
    )


if __name__ == "__main__":
    main()
