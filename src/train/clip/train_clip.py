"""
CLIP Training & Inference - custom training loop with GradientTape.

Implements contrastive language-image pre-training with CLIP-specific metrics
(accuracy, recall@k), robust checkpointing, and zero-shot retrieval inference.

Reference: Radford et al. (2021) "Learning Transferable Visual Models from
Natural Language Supervision" https://arxiv.org/abs/2103.00020
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np
import keras
from keras import ops
import tensorflow as tf

from train.common import setup_gpu
from dl_techniques.utils.logger import logger
from dl_techniques.optimization import (
    optimizer_builder,
    learning_rate_schedule_builder
)
from dl_techniques.layers.tokenizers.bpe import BPETokenizer
from dl_techniques.models.clip.model import create_clip_model
from dl_techniques.losses.clip_contrastive_loss import CLIPContrastiveLoss
from dl_techniques.metrics.clip_accuracy import CLIPAccuracy, CLIPRecallAtK


# ---------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------

class CLIPDataLoader:
    """Data loader for CLIP training with image preprocessing and text tokenization.

    Args:
        image_size: Size to resize images to (square).
        context_length: Maximum length for text sequences.
        tokenizer: Text tokenizer instance (e.g., BPETokenizer).
        image_normalization: Normalization strategy ('imagenet', 'clip', or None).
        augmentation: Whether to apply data augmentation during training.
    """

    def __init__(
        self,
        image_size: int = 224,
        context_length: int = 77,
        tokenizer: Optional[Any] = None,
        image_normalization: str = 'clip',
        augmentation: bool = False
    ) -> None:
        self.image_size = image_size
        self.context_length = context_length
        self.tokenizer = tokenizer or self._create_simple_tokenizer()
        self.image_normalization = image_normalization
        self.augmentation = augmentation
        logger.info(
            f"CLIPDataLoader: image_size={image_size}, context_length={context_length}, "
            f"normalization={image_normalization}, augmentation={augmentation}"
        )

    def _create_simple_tokenizer(self) -> Dict[str, int]:
        """Create a simple vocabulary for demo purposes."""
        vocab = {f"word_{i}": i for i in range(1000)}
        vocab.update({"<pad>": 0, "<unk>": 1, "<start>": 2, "<end>": 3})
        logger.warning("Using simple tokenizer for demo. For production, use BPETokenizer.")
        return vocab

    def preprocess_image(self, image_path: str, training: bool = True) -> tf.Tensor:
        """Load and preprocess image from file path."""
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.cast(image, tf.float32)

        if training and self.augmentation:
            image = self._augment_image(image)

        image = tf.image.resize(
            image, (self.image_size, self.image_size),
            method='bilinear', antialias=True
        )

        if self.image_normalization == 'imagenet':
            mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
            std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
            image = (image / 255.0 - mean) / std
        elif self.image_normalization == 'clip':
            image = image / 255.0

        return image

    def _augment_image(self, image: tf.Tensor) -> tf.Tensor:
        """Apply data augmentation to image."""
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        if image.shape[-1] == 3:
            image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
            image = tf.image.random_hue(image, max_delta=0.05)
        return tf.clip_by_value(image, 0.0, 255.0)

    def preprocess_text(self, text: str) -> tf.Tensor:
        """Tokenize text into token IDs of shape (context_length,)."""
        if isinstance(self.tokenizer, dict):
            words = text.lower().split()[:self.context_length - 2]
            token_ids = [self.tokenizer.get(word, 1) for word in words]
        else:
            token_ids = self.tokenizer.encode(text)[:self.context_length]

        # Pad or truncate
        token_ids = token_ids[:self.context_length]
        if len(token_ids) < self.context_length:
            token_ids = token_ids + [0] * (self.context_length - len(token_ids))

        return tf.constant(token_ids, dtype=tf.int32)

    def create_dataset(
        self,
        image_paths: List[str],
        captions: List[str],
        batch_size: int = 32,
        shuffle: bool = True,
        buffer_size: Optional[int] = None,
        num_parallel_calls: int = tf.data.AUTOTUNE,
        prefetch_size: int = tf.data.AUTOTUNE,
        drop_remainder: bool = True
    ) -> tf.data.Dataset:
        """Create batched tf.data.Dataset for CLIP training."""
        if len(image_paths) != len(captions):
            raise ValueError(
                f"Number of images ({len(image_paths)}) must match captions ({len(captions)})"
            )

        logger.info(f"Creating dataset: {len(image_paths)} samples, batch_size={batch_size}")

        dataset = tf.data.Dataset.from_tensor_slices((image_paths, captions))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=buffer_size or len(image_paths))

        def preprocess_fn(img_path, caption):
            image = self.preprocess_image(img_path, training=shuffle)
            text = self.preprocess_text(caption)
            return image, text

        dataset = dataset.map(preprocess_fn, num_parallel_calls=num_parallel_calls)
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
        dataset = dataset.prefetch(prefetch_size)
        return dataset


# ---------------------------------------------------------------------
# Training Callbacks
# ---------------------------------------------------------------------

class CLIPCheckpoint(keras.callbacks.Callback):
    """Checkpoint callback that saves models based on validation metrics.

    Args:
        filepath: Path to save checkpoint (can include formatting).
        monitor: Metric to monitor for best model selection.
        mode: 'min' or 'max' for metric monitoring.
        save_best_only: Whether to save only the best model.
        save_freq: Frequency of saving ('epoch' or integer).
        verbose: Verbosity level.
    """

    def __init__(
        self, filepath: str, monitor: str = 'val_loss', mode: str = 'min',
        save_best_only: bool = True, save_freq: Union[str, int] = 'epoch',
        verbose: int = 1
    ) -> None:
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        self.verbose = verbose

        if mode == 'min':
            self.best_metric = np.inf
            self.monitor_op = np.less
        else:
            self.best_metric = -np.inf
            self.monitor_op = np.greater

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None) -> None:
        if logs is None or self.save_freq != 'epoch':
            return

        current = logs.get(self.monitor)
        if current is None:
            logger.warning(f"Metric '{self.monitor}' not found. Available: {list(logs.keys())}")
            return

        should_save = False
        if self.save_best_only:
            if self.monitor_op(current, self.best_metric):
                should_save = True
                self.best_metric = current
                if self.verbose > 0:
                    logger.info(f"Epoch {epoch + 1}: {self.monitor} improved to {current:.4f}")
        else:
            should_save = True

        if should_save:
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            self.model.save(filepath)
            if self.verbose > 0:
                logger.info(f"Saved checkpoint to {filepath}")


class LearningRateLogger(keras.callbacks.Callback):
    """Callback to log learning rate at each epoch."""

    def __init__(self, verbose: int = 1) -> None:
        super().__init__()
        self.verbose = verbose

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None) -> None:
        if logs is None:
            logs = {}

        if hasattr(self.model, 'optimizer') and self.model.optimizer is not None:
            lr = self.model.optimizer.learning_rate
            if hasattr(lr, 'numpy'):
                lr_value = float(lr.numpy())
            elif callable(lr):
                lr_value = float(lr(self.model.optimizer.iterations))
            else:
                lr_value = float(lr)

            logs['learning_rate'] = lr_value
            if self.verbose > 0:
                logger.info(f"Epoch {epoch + 1}: learning_rate = {lr_value:.2e}")


# ---------------------------------------------------------------------
# Training Class
# ---------------------------------------------------------------------

class CLIPTrainer:
    """Custom training loop for CLIP using GradientTape.

    Supports multiple metrics, validation, LR scheduling, checkpointing,
    mixed precision, and gradient clipping.

    Args:
        model: CLIP model instance.
        loss_fn: Loss function (CLIPContrastiveLoss).
        optimizer: Optimizer instance.
        metrics: List of metric instances to track.
        gradient_clip_norm: Optional gradient clipping value.
        mixed_precision: Whether to use mixed precision training.
    """

    def __init__(
        self,
        model: Any,
        loss_fn: keras.losses.Loss,
        optimizer: keras.optimizers.Optimizer,
        metrics: Optional[List[keras.metrics.Metric]] = None,
        gradient_clip_norm: Optional[float] = None,
        mixed_precision: bool = False
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metrics = metrics or []
        self.gradient_clip_norm = gradient_clip_norm
        self.mixed_precision = mixed_precision

        self.current_epoch = 0
        self.global_step = 0
        self.training_history = {'loss': [], 'val_loss': [], 'learning_rate': []}
        for metric in self.metrics:
            self.training_history[metric.name] = []
            self.training_history[f'val_{metric.name}'] = []

        if mixed_precision:
            keras.mixed_precision.set_global_policy(
                keras.mixed_precision.Policy('mixed_float16')
            )
            logger.info("Mixed precision training enabled")

        logger.info(
            f"CLIPTrainer: {len(self.metrics)} metrics, "
            f"gradient_clip={gradient_clip_norm}, mixed_precision={mixed_precision}"
        )

    @tf.function
    def train_step(self, batch: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Execute single training step."""
        images, texts = batch
        inputs = {'image': images, 'text': texts}

        with tf.GradientTape() as tape:
            outputs = self.model(inputs, training=True)
            loss = self.loss_fn(None, outputs)
            if self.model.losses:
                loss += tf.add_n(self.model.losses)
            if self.mixed_precision:
                loss = self.optimizer.get_scaled_loss(loss)

        gradients = tape.gradient(loss, self.model.trainable_variables)

        if self.mixed_precision:
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        if self.gradient_clip_norm is not None:
            gradients, _ = tf.clip_by_global_norm(gradients, self.gradient_clip_norm)

        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        for metric in self.metrics:
            metric.update_state(outputs)

        return {'loss': loss}

    @tf.function
    def val_step(self, batch: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Execute single validation step."""
        images, texts = batch
        inputs = {'image': images, 'text': texts}
        outputs = self.model(inputs, training=False)
        loss = self.loss_fn(None, outputs)

        for metric in self.metrics:
            metric.update_state(outputs)

        return {'val_loss': loss}

    def fit(
        self,
        train_dataset: tf.data.Dataset,
        validation_dataset: Optional[tf.data.Dataset] = None,
        epochs: int = 10,
        steps_per_epoch: Optional[int] = None,
        validation_steps: Optional[int] = None,
        callbacks: Optional[List[keras.callbacks.Callback]] = None,
        verbose: int = 1,
        initial_epoch: int = 0
    ) -> Dict[str, List[float]]:
        """Train the CLIP model."""
        logger.info(f"Starting training for {epochs} epochs")
        callbacks = callbacks or []

        for callback in callbacks:
            callback.set_model(self)
            if hasattr(callback, 'on_train_begin'):
                callback.on_train_begin()

        try:
            for epoch in range(initial_epoch, epochs):
                self.current_epoch = epoch
                if verbose >= 1:
                    logger.info(f"\nEpoch {epoch + 1}/{epochs}")

                for callback in callbacks:
                    if hasattr(callback, 'on_epoch_begin'):
                        callback.on_epoch_begin(epoch)

                train_metrics = self._train_epoch(train_dataset, steps_per_epoch, verbose)

                val_metrics = {}
                if validation_dataset is not None:
                    val_metrics = self._validate_epoch(validation_dataset, validation_steps, verbose)

                epoch_logs = {**train_metrics, **val_metrics}
                for key, value in epoch_logs.items():
                    if key in self.training_history:
                        self.training_history[key].append(float(value))

                if verbose >= 1:
                    log_str = f"Epoch {epoch + 1}/{epochs} - "
                    log_str += " - ".join(f"{k}: {v:.4f}" for k, v in epoch_logs.items())
                    logger.info(log_str)

                for callback in callbacks:
                    if hasattr(callback, 'on_epoch_end'):
                        callback.on_epoch_end(epoch, epoch_logs)

        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
        finally:
            for callback in callbacks:
                if hasattr(callback, 'on_train_end'):
                    callback.on_train_end()

        logger.info("Training completed!")
        return self.training_history

    def _train_epoch(
        self, dataset: tf.data.Dataset, steps_per_epoch: Optional[int], verbose: int
    ) -> Dict[str, float]:
        """Execute one training epoch."""
        for metric in self.metrics:
            metric.reset_state()

        loss_sum = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataset):
            if steps_per_epoch and batch_idx >= steps_per_epoch:
                break

            step_results = self.train_step(batch)
            loss_sum += float(step_results['loss'])
            num_batches += 1
            self.global_step += 1

            if verbose >= 2 and batch_idx % 100 == 0:
                logger.info(f"  Batch {batch_idx}: loss={float(step_results['loss']):.4f}")

        metrics_dict = {'loss': loss_sum / max(num_batches, 1)}
        for metric in self.metrics:
            metrics_dict[metric.name] = float(metric.result())
        return metrics_dict

    def _validate_epoch(
        self, dataset: tf.data.Dataset, validation_steps: Optional[int], verbose: int
    ) -> Dict[str, float]:
        """Execute one validation epoch."""
        for metric in self.metrics:
            metric.reset_state()

        loss_sum = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataset):
            if validation_steps and batch_idx >= validation_steps:
                break
            step_results = self.val_step(batch)
            loss_sum += float(step_results['val_loss'])
            num_batches += 1

        metrics_dict = {'val_loss': loss_sum / max(num_batches, 1)}
        for metric in self.metrics:
            metrics_dict[f'val_{metric.name}'] = float(metric.result())
        return metrics_dict

    def save_model(self, filepath: str) -> None:
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")

    def save_history(self, filepath: str) -> None:
        with open(filepath, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        logger.info(f"Training history saved to {filepath}")


# ---------------------------------------------------------------------
# Inference Utilities
# ---------------------------------------------------------------------

class CLIPInference:
    """Inference utilities for trained CLIP models.

    Provides image/text encoding, similarity computation, and zero-shot retrieval.

    Args:
        model: Trained CLIP model instance.
        data_loader: Data loader instance for preprocessing.
        batch_size: Batch size for inference.
    """

    def __init__(self, model: Any, data_loader: CLIPDataLoader, batch_size: int = 64) -> None:
        self.model = model
        self.data_loader = data_loader
        self.batch_size = batch_size

    def encode_images(self, image_paths: List[str], normalize: bool = True, show_progress: bool = True) -> np.ndarray:
        """Encode images to embeddings of shape (num_images, embed_dim)."""
        embeddings_list = []
        num_images = len(image_paths)
        logger.info(f"Encoding {num_images} images...")

        for i in range(0, num_images, self.batch_size):
            batch_paths = image_paths[i:i + self.batch_size]
            batch_images = []
            for path in batch_paths:
                try:
                    image = self.data_loader.preprocess_image(path, training=False)
                    batch_images.append(image.numpy())
                except Exception as e:
                    logger.error(f"Error loading image {path}: {e}")
                    batch_images.append(
                        np.zeros((self.data_loader.image_size, self.data_loader.image_size, 3))
                    )

            batch_tensor = ops.convert_to_tensor(np.array(batch_images))
            batch_embeddings = self.model.encode_image(batch_tensor, training=False)
            embeddings_list.append(ops.convert_to_numpy(batch_embeddings))

            if show_progress and (i // self.batch_size) % 10 == 0:
                logger.info(f"  Processed {i + len(batch_paths)}/{num_images} images")

        embeddings = np.vstack(embeddings_list)
        if normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings

    def encode_texts(self, texts: List[str], normalize: bool = True, show_progress: bool = True) -> np.ndarray:
        """Encode texts to embeddings of shape (num_texts, embed_dim)."""
        embeddings_list = []
        num_texts = len(texts)
        logger.info(f"Encoding {num_texts} texts...")

        for i in range(0, num_texts, self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_tokens = []
            for text in batch_texts:
                try:
                    tokens = self.data_loader.preprocess_text(text)
                    batch_tokens.append(tokens.numpy())
                except Exception as e:
                    logger.error(f"Error tokenizing text '{text[:50]}...': {e}")
                    batch_tokens.append(np.zeros(self.data_loader.context_length, dtype=np.int32))

            batch_tensor = ops.convert_to_tensor(np.array(batch_tokens))
            batch_embeddings = self.model.encode_text(batch_tensor, training=False)
            embeddings_list.append(ops.convert_to_numpy(batch_embeddings))

            if show_progress and (i // self.batch_size) % 10 == 0:
                logger.info(f"  Processed {i + len(batch_texts)}/{num_texts} texts")

        embeddings = np.vstack(embeddings_list)
        if normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings

    def compute_similarity(self, image_embeddings: np.ndarray, text_embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity matrix between image and text embeddings."""
        image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
        text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
        return np.dot(image_embeddings, text_embeddings.T)

    def retrieve_images(self, query_text: str, image_paths: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve top-k images for a text query."""
        text_embedding = self.encode_texts([query_text], show_progress=False)
        image_embeddings = self.encode_images(image_paths, show_progress=True)
        similarities = self.compute_similarity(image_embeddings, text_embedding)
        scores = similarities[:, 0]
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = [(image_paths[i], float(scores[i])) for i in top_indices]
        for rank, (path, score) in enumerate(results, 1):
            logger.info(f"  {rank}. {Path(path).name}: {score:.4f}")
        return results

    def retrieve_texts(self, query_image_path: str, texts: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve top-k texts for an image query."""
        image_embedding = self.encode_images([query_image_path], show_progress=False)
        text_embeddings = self.encode_texts(texts, show_progress=True)
        similarities = self.compute_similarity(image_embedding, text_embeddings)
        scores = similarities[0, :]
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = [(texts[i], float(scores[i])) for i in top_indices]
        for rank, (text, score) in enumerate(results, 1):
            text_preview = text[:50] + "..." if len(text) > 50 else text
            logger.info(f"  {rank}. '{text_preview}': {score:.4f}")
        return results


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def _create_synthetic_clip_data(num_samples: int = 64, image_size: int = 224):
    """Create synthetic image/caption data for validation runs."""
    import tempfile
    from PIL import Image

    tmp_dir = tempfile.mkdtemp(prefix="clip_synthetic_")
    image_paths = []
    captions = []
    sample_captions = [
        "a photo of a cat", "a photo of a dog", "a red car",
        "a blue sky", "a green tree", "a white house",
        "a black cat", "a brown dog", "a yellow flower",
        "a purple sunset", "a silver plane", "an orange fruit",
    ]

    for i in range(num_samples):
        img = Image.fromarray(np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8))
        path = os.path.join(tmp_dir, f"img_{i:04d}.png")
        img.save(path)
        image_paths.append(path)
        captions.append(sample_captions[i % len(sample_captions)])

    return image_paths, captions


def main():
    """Main entry point for CLIP training."""
    parser = argparse.ArgumentParser(description="Train CLIP model")
    parser.add_argument("--gpu", type=int, default=None, help="GPU device index")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--image-dir", type=str, default=None,
                        help="Directory with images (uses synthetic data if not provided)")
    parser.add_argument("--captions-file", type=str, default=None,
                        help="Path to captions file (one caption per line, matching image order)")
    args = parser.parse_args()

    setup_gpu(gpu_id=args.gpu)

    # Create model
    model = create_clip_model()
    logger.info(f"Created CLIP model: {model.count_params():,} parameters")

    # Create training components
    loss_fn = CLIPContrastiveLoss()
    lr_schedule = learning_rate_schedule_builder({
        "type": "cosine_decay", "warmup_steps": 1000, "warmup_start_lr": 1e-8,
        "learning_rate": args.learning_rate, "decay_steps": 50000, "alpha": 0.0001
    })
    optimizer = optimizer_builder({
        "type": "adamw", "beta_1": 0.9, "beta_2": 0.999,
        "gradient_clipping_by_norm": 1.0
    }, lr_schedule)

    metrics = [CLIPAccuracy(), CLIPRecallAtK(k=5)]

    trainer = CLIPTrainer(
        model=model, loss_fn=loss_fn, optimizer=optimizer,
        metrics=metrics, gradient_clip_norm=1.0
    )

    # Load or create dataset
    if args.image_dir and args.captions_file:
        import glob
        image_paths = sorted(glob.glob(os.path.join(args.image_dir, "*.jpg")) +
                             glob.glob(os.path.join(args.image_dir, "*.png")))
        with open(args.captions_file, 'r') as f:
            captions = [line.strip() for line in f.readlines()]
        logger.info(f"Loaded {len(image_paths)} images and {len(captions)} captions")
    else:
        logger.info("No dataset provided, using synthetic data for validation")
        image_paths, captions = _create_synthetic_clip_data(
            num_samples=64, image_size=224,
        )

    data_loader = CLIPDataLoader(image_size=224, context_length=77)
    train_dataset = data_loader.create_dataset(
        image_paths, captions, batch_size=args.batch_size,
    )

    logger.info(f"Starting CLIP training: {args.epochs} epochs, batch_size={args.batch_size}")
    trainer.fit(train_dataset, epochs=args.epochs)


if __name__ == "__main__":
    main()
