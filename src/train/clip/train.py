"""
Modern CLIP Training & Inference Flow - Refined Version
========================================================

This module implements a production-ready training loop and inference utilities
for the CLIP (Contrastive Language-Image Pre-training) model using Keras 3 and
following dl-techniques project best practices.

Key Improvements:
-----------------
1. CLIP-specific metrics (accuracy, recall@k for both directions)
2. Robust training callbacks (checkpointing, LR monitoring, early stopping)
3. Better data loading with proper preprocessing
4. Comprehensive evaluation utilities
5. Proper logging using project logger
6. Configuration management
7. Support for mixed precision training
8. Batch-wise and epoch-wise metric tracking

Training Flow:
--------------
1. **Data Preparation**: Load and preprocess image-text pairs
2. **Model Setup**: Create CLIP model with specified architecture
3. **Training**: Use custom training loop with TensorFlow GradientTape
4. **Validation**: Evaluate on validation set with CLIP-specific metrics
5. **Checkpointing**: Save best models based on validation metrics
6. **Inference**: Zero-shot retrieval and feature extraction

References:
-----------
- Radford, A., et al. (2021). Learning Transferable Visual Models from
  Natural Language Supervision. ICML. https://arxiv.org/abs/2103.00020
"""

import json
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np
import keras
from keras import ops
import tensorflow as tf

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

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
    """
    Enhanced data loader for CLIP training with proper preprocessing.

    Handles image loading, preprocessing, and text tokenization for CLIP
    training. Supports various image formats and efficient batching.

    Args:
        image_size: Size to resize images to (square).
        context_length: Maximum length for text sequences.
        tokenizer: Text tokenizer instance (e.g., BPETokenizer).
        image_normalization: Normalization strategy for images.
            Options: 'imagenet', 'clip', or None.
        augmentation: Whether to apply data augmentation during training.

    Example:
        ```python
        tokenizer = BPETokenizer(vocab_file='vocab.json')
        loader = CLIPDataLoader(
            image_size=224,
            context_length=77,
            tokenizer=tokenizer,
            augmentation=True
        )

        # Create dataset
        dataset = loader.create_dataset(
            image_paths=train_images,
            captions=train_captions,
            batch_size=32,
            shuffle=True
        )
        ```
    """

    def __init__(
        self,
        image_size: int = 224,
        context_length: int = 77,
        tokenizer: Optional[Any] = None,
        image_normalization: str = 'clip',
        augmentation: bool = False
    ) -> None:
        """Initialize CLIP data loader."""
        self.image_size = image_size
        self.context_length = context_length
        self.tokenizer = tokenizer or self._create_simple_tokenizer()
        self.image_normalization = image_normalization
        self.augmentation = augmentation

        logger.info(
            f"CLIPDataLoader initialized: image_size={image_size}, "
            f"context_length={context_length}, normalization={image_normalization}, "
            f"augmentation={augmentation}"
        )

    def _create_simple_tokenizer(self) -> Dict[str, int]:
        """Create a simple vocabulary for demo purposes."""
        vocab = {f"word_{i}": i for i in range(1000)}
        vocab["<pad>"] = 0
        vocab["<unk>"] = 1
        vocab["<start>"] = 2
        vocab["<end>"] = 3
        logger.warning(
            "Using simple tokenizer for demo. For production, use BPETokenizer."
        )
        return vocab

    def preprocess_image(
        self,
        image_path: str,
        training: bool = True
    ) -> tf.Tensor:
        """
        Preprocess image from file path.

        Args:
            image_path: Path to image file.
            training: Whether in training mode (affects augmentation).

        Returns:
            Preprocessed image tensor of shape (image_size, image_size, 3).
        """
        # Load and decode image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.cast(image, tf.float32)

        # Apply augmentation if in training mode
        if training and self.augmentation:
            image = self._augment_image(image)

        # Resize to target size
        image = tf.image.resize(
            image,
            (self.image_size, self.image_size),
            method='bilinear',
            antialias=True
        )

        # Normalize
        if self.image_normalization == 'imagenet':
            # ImageNet mean and std
            mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
            std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
            image = image / 255.0
            image = (image - mean) / std
        elif self.image_normalization == 'clip':
            # CLIP normalization: scale to [0, 1]
            image = image / 255.0
        else:
            # No normalization, assume image is already in correct range
            pass

        return image

    def _augment_image(self, image: tf.Tensor) -> tf.Tensor:
        """
        Apply data augmentation to image.

        Args:
            image: Input image tensor.

        Returns:
            Augmented image tensor.
        """
        # Random horizontal flip
        image = tf.image.random_flip_left_right(image)

        # Random brightness and contrast
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

        # Random saturation and hue (only if image has 3 channels)
        if image.shape[-1] == 3:
            image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
            image = tf.image.random_hue(image, max_delta=0.05)

        # Clip values to valid range
        image = tf.clip_by_value(image, 0.0, 255.0)

        return image

    def preprocess_text(self, text: str) -> tf.Tensor:
        """
        Preprocess text into token IDs.

        Args:
            text: Input text string.

        Returns:
            Token ID tensor of shape (context_length,).
        """
        # Tokenize using provided tokenizer
        if isinstance(self.tokenizer, dict):
            # Simple dictionary tokenizer
            words = text.lower().split()[:self.context_length - 2]
            token_ids = [self.tokenizer.get(word, 1) for word in words]
        else:
            # Assume BPETokenizer or similar with encode method
            token_ids = self.tokenizer.encode(text)[:self.context_length]

        # Pad or truncate to context_length
        if len(token_ids) < self.context_length:
            # Pad with zeros
            padding = [0] * (self.context_length - len(token_ids))
            token_ids = token_ids + padding
        else:
            # Truncate
            token_ids = token_ids[:self.context_length]

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
        """
        Create TensorFlow dataset for CLIP training.

        Args:
            image_paths: List of paths to image files.
            captions: List of text captions corresponding to images.
            batch_size: Batch size for training.
            shuffle: Whether to shuffle the dataset.
            buffer_size: Shuffle buffer size. If None, uses len(image_paths).
            num_parallel_calls: Parallel calls for preprocessing.
            prefetch_size: Number of batches to prefetch.
            drop_remainder: Whether to drop incomplete batches.

        Returns:
            Batched and preprocessed tf.data.Dataset.
        """
        if len(image_paths) != len(captions):
            raise ValueError(
                f"Number of images ({len(image_paths)}) must match "
                f"number of captions ({len(captions)})"
            )

        logger.info(
            f"Creating dataset with {len(image_paths)} samples, "
            f"batch_size={batch_size}, shuffle={shuffle}"
        )

        # Create dataset from file paths and captions
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, captions))

        # Shuffle if requested
        if shuffle:
            if buffer_size is None:
                buffer_size = len(image_paths)
            dataset = dataset.shuffle(buffer_size=buffer_size)

        # Map preprocessing function
        def preprocess_fn(img_path, caption):
            image = self.preprocess_image(img_path, training=shuffle)
            text = self.preprocess_text(caption)
            return image, text

        dataset = dataset.map(
            preprocess_fn,
            num_parallel_calls=num_parallel_calls
        )

        # Batch and prefetch
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
        dataset = dataset.prefetch(prefetch_size)

        return dataset


# ---------------------------------------------------------------------
# Training Callbacks
# ---------------------------------------------------------------------


class CLIPCheckpoint(keras.callbacks.Callback):
    """
    Custom checkpoint callback for CLIP training.

    Saves model checkpoints based on validation metrics. Supports saving
    best model only or periodic checkpoints.

    Args:
        filepath: Path to save checkpoint (can include formatting).
        monitor: Metric to monitor for best model selection.
        mode: 'min' or 'max' for metric monitoring.
        save_best_only: Whether to save only the best model.
        save_freq: Frequency of saving ('epoch' or integer for batch-level).
        verbose: Verbosity level.

    Example:
        ```python
        checkpoint = CLIPCheckpoint(
            filepath='checkpoints/clip_epoch_{epoch:02d}.keras',
            monitor='val_loss',
            mode='min',
            save_best_only=True
        )

        trainer.fit(train_dataset, validation_dataset, callbacks=[checkpoint])
        ```
    """

    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        save_freq: Union[str, int] = 'epoch',
        verbose: int = 1
    ) -> None:
        """Initialize checkpoint callback."""
        super().__init__()

        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        self.verbose = verbose

        # Initialize best metric
        if mode == 'min':
            self.best_metric = np.inf
            self.monitor_op = np.less
        else:
            self.best_metric = -np.inf
            self.monitor_op = np.greater

        # Create checkpoint directory
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        logger.info(
            f"CLIPCheckpoint initialized: monitor={monitor}, mode={mode}, "
            f"save_best_only={save_best_only}"
        )

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None) -> None:
        """Save checkpoint at end of epoch if conditions are met."""
        if logs is None:
            logs = {}

        if self.save_freq != 'epoch':
            return

        # Get current metric value
        current = logs.get(self.monitor)

        if current is None:
            logger.warning(
                f"Checkpoint monitoring metric '{self.monitor}' not found in logs. "
                f"Available metrics: {list(logs.keys())}"
            )
            return

        # Determine if should save
        should_save = False
        if self.save_best_only:
            if self.monitor_op(current, self.best_metric):
                should_save = True
                self.best_metric = current
                if self.verbose > 0:
                    logger.info(
                        f"Epoch {epoch + 1}: {self.monitor} improved to {current:.4f}"
                    )
        else:
            should_save = True

        # Save model
        if should_save:
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            self.model.save(filepath)
            if self.verbose > 0:
                logger.info(f"Saved checkpoint to {filepath}")


class LearningRateLogger(keras.callbacks.Callback):
    """
    Callback to log learning rate at each epoch.

    Logs the current learning rate to the training history and console.
    Useful for monitoring learning rate schedules.

    Args:
        verbose: Verbosity level (0 = silent, 1 = progress messages).

    Example:
        ```python
        lr_logger = LearningRateLogger(verbose=1)
        trainer.fit(train_dataset, callbacks=[lr_logger])
        ```
    """

    def __init__(self, verbose: int = 1) -> None:
        """Initialize learning rate logger."""
        super().__init__()
        self.verbose = verbose

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None) -> None:
        """Log learning rate at end of epoch."""
        if logs is None:
            logs = {}

        # Get current learning rate
        if hasattr(self.model, 'optimizer') and self.model.optimizer is not None:
            lr = self.model.optimizer.learning_rate

            # Handle different LR types
            if hasattr(lr, 'numpy'):
                lr_value = float(lr.numpy())
            elif callable(lr):
                # Learning rate schedule
                lr_value = float(lr(self.model.optimizer.iterations))
            else:
                lr_value = float(lr)

            # Store in logs
            logs['learning_rate'] = lr_value

            if self.verbose > 0:
                logger.info(f"Epoch {epoch + 1}: learning_rate = {lr_value:.2e}")


# ---------------------------------------------------------------------
# Training Class
# ---------------------------------------------------------------------


class CLIPTrainer:
    """
    Enhanced trainer class for CLIP models with comprehensive metrics.

    Implements custom training loop using TensorFlow GradientTape with
    support for:
    - Multiple metrics (accuracy, recall@k)
    - Validation with early stopping
    - Learning rate scheduling
    - Model checkpointing
    - Mixed precision training
    - Gradient clipping

    Args:
        model: CLIP model instance.
        loss_fn: Loss function (CLIPContrastiveLoss).
        optimizer: Optimizer instance.
        metrics: List of metric instances to track.
        gradient_clip_norm: Optional gradient clipping value.
        mixed_precision: Whether to use mixed precision training.

    Example:
        ```python
        # Create model and training components
        model = create_clip_variant("ViT-B/32")
        loss_fn = CLIPContrastiveLoss(temperature=0.07, apply_temperature=True)
        optimizer = optimizer_builder(opt_config, lr_schedule)

        # Create metrics
        metrics = [
            CLIPAccuracy(direction='i2t'),
            CLIPAccuracy(direction='t2i'),
            CLIPRecallAtK(k=5, direction='i2t'),
            CLIPRecallAtK(k=5, direction='t2i'),
        ]

        # Create trainer
        trainer = CLIPTrainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            metrics=metrics,
            gradient_clip_norm=1.0
        )

        # Train
        history = trainer.fit(
            train_dataset=train_ds,
            validation_dataset=val_ds,
            epochs=30,
            callbacks=[checkpoint, lr_logger]
        )
        ```
    """

    def __init__(
        self,
        model: Any,  # CLIP model
        loss_fn: keras.losses.Loss,
        optimizer: keras.optimizers.Optimizer,
        metrics: Optional[List[keras.metrics.Metric]] = None,
        gradient_clip_norm: Optional[float] = None,
        mixed_precision: bool = False
    ) -> None:
        """Initialize CLIP trainer."""
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metrics = metrics or []
        self.gradient_clip_norm = gradient_clip_norm
        self.mixed_precision = mixed_precision

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.training_history = {
            'loss': [],
            'val_loss': [],
            'learning_rate': []
        }

        # Add metric names to history
        for metric in self.metrics:
            self.training_history[metric.name] = []
            self.training_history[f'val_{metric.name}'] = []

        # Setup mixed precision if enabled
        if mixed_precision:
            policy = keras.mixed_precision.Policy('mixed_float16')
            keras.mixed_precision.set_global_policy(policy)
            logger.info("Mixed precision training enabled")

        logger.info(
            f"CLIPTrainer initialized with {len(self.metrics)} metrics, "
            f"gradient_clip_norm={gradient_clip_norm}, "
            f"mixed_precision={mixed_precision}"
        )

    @tf.function
    def train_step(
        self,
        batch: Tuple[tf.Tensor, tf.Tensor]
    ) -> Dict[str, tf.Tensor]:
        """
        Execute single training step.

        Args:
            batch: Tuple of (images, texts).

        Returns:
            Dictionary of metric values for this batch.
        """
        images, texts = batch
        inputs = {'image': images, 'text': texts}

        with tf.GradientTape() as tape:
            # Forward pass
            outputs = self.model(inputs, training=True)

            # Compute loss (pass dummy y_true as loss generates labels internally)
            loss = self.loss_fn(None, outputs)

            # Add regularization losses if any
            if self.model.losses:
                loss += tf.add_n(self.model.losses)

            # Scale loss for mixed precision
            if self.mixed_precision:
                loss = self.optimizer.get_scaled_loss(loss)

        # Compute gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)

        # Unscale gradients for mixed precision
        if self.mixed_precision:
            gradients = self.optimizer.get_unscaled_gradients(gradients)

        # Clip gradients if requested
        if self.gradient_clip_norm is not None:
            gradients, _ = tf.clip_by_global_norm(
                gradients, self.gradient_clip_norm
            )

        # Apply gradients
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables)
        )

        # Update metrics
        results = {'loss': loss}
        for metric in self.metrics:
            metric.update_state(outputs)

        return results

    @tf.function
    def val_step(
        self,
        batch: Tuple[tf.Tensor, tf.Tensor]
    ) -> Dict[str, tf.Tensor]:
        """
        Execute single validation step.

        Args:
            batch: Tuple of (images, texts).

        Returns:
            Dictionary of metric values for this batch.
        """
        images, texts = batch
        inputs = {'image': images, 'text': texts}

        # Forward pass (no gradient tracking)
        outputs = self.model(inputs, training=False)

        # Compute loss
        loss = self.loss_fn(None, outputs)

        # Update metrics
        results = {'val_loss': loss}
        for metric in self.metrics:
            metric.update_state(outputs)

        return results

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
        """
        Train the CLIP model.

        Args:
            train_dataset: Training dataset.
            validation_dataset: Optional validation dataset.
            epochs: Number of epochs to train.
            steps_per_epoch: Optional limit on training steps per epoch.
            validation_steps: Optional limit on validation steps.
            callbacks: List of callbacks to apply during training.
            verbose: Verbosity level (0 = silent, 1 = progress, 2 = detailed).
            initial_epoch: Epoch to start training from (for resuming).

        Returns:
            Training history dictionary with metric values.
        """
        logger.info(f"Starting training for {epochs} epochs")

        callbacks = callbacks or []

        # Initialize callbacks
        for callback in callbacks:
            callback.set_model(self)
            if hasattr(callback, 'on_train_begin'):
                callback.on_train_begin()

        try:
            for epoch in range(initial_epoch, epochs):
                self.current_epoch = epoch

                if verbose >= 1:
                    logger.info(f"\nEpoch {epoch + 1}/{epochs}")

                # Call epoch begin callbacks
                for callback in callbacks:
                    if hasattr(callback, 'on_epoch_begin'):
                        callback.on_epoch_begin(epoch)

                # Training phase
                train_metrics = self._train_epoch(
                    train_dataset,
                    steps_per_epoch,
                    verbose
                )

                # Validation phase
                val_metrics = {}
                if validation_dataset is not None:
                    val_metrics = self._validate_epoch(
                        validation_dataset,
                        validation_steps,
                        verbose
                    )

                # Combine metrics for logging
                epoch_logs = {**train_metrics, **val_metrics}

                # Update history
                for key, value in epoch_logs.items():
                    if key in self.training_history:
                        self.training_history[key].append(float(value))

                # Log epoch results
                if verbose >= 1:
                    log_str = f"Epoch {epoch + 1}/{epochs} - "
                    log_str += " - ".join([
                        f"{key}: {value:.4f}"
                        for key, value in epoch_logs.items()
                    ])
                    logger.info(log_str)

                # Call epoch end callbacks
                for callback in callbacks:
                    if hasattr(callback, 'on_epoch_end'):
                        callback.on_epoch_end(epoch, epoch_logs)

        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")

        finally:
            # Call training end callbacks
            for callback in callbacks:
                if hasattr(callback, 'on_train_end'):
                    callback.on_train_end()

        logger.info("Training completed!")
        return self.training_history

    def _train_epoch(
        self,
        dataset: tf.data.Dataset,
        steps_per_epoch: Optional[int],
        verbose: int
    ) -> Dict[str, float]:
        """
        Execute one training epoch.

        Args:
            dataset: Training dataset.
            steps_per_epoch: Optional step limit.
            verbose: Verbosity level.

        Returns:
            Dictionary of epoch metrics.
        """
        # Reset metrics
        for metric in self.metrics:
            metric.reset_state()

        # Training loop
        loss_sum = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataset):
            if steps_per_epoch and batch_idx >= steps_per_epoch:
                break

            # Execute training step
            step_results = self.train_step(batch)
            loss_val = float(step_results['loss'])
            loss_sum += loss_val
            num_batches += 1
            self.global_step += 1

            # Log batch progress
            if verbose >= 2 and batch_idx % 100 == 0:
                logger.info(f"  Batch {batch_idx}: loss={loss_val:.4f}")

        # Compute epoch metrics
        avg_loss = loss_sum / max(num_batches, 1)
        metrics_dict = {'loss': avg_loss}

        for metric in self.metrics:
            metrics_dict[metric.name] = float(metric.result())

        return metrics_dict

    def _validate_epoch(
        self,
        dataset: tf.data.Dataset,
        validation_steps: Optional[int],
        verbose: int
    ) -> Dict[str, float]:
        """
        Execute one validation epoch.

        Args:
            dataset: Validation dataset.
            validation_steps: Optional step limit.
            verbose: Verbosity level.

        Returns:
            Dictionary of validation metrics.
        """
        # Reset metrics
        for metric in self.metrics:
            metric.reset_state()

        # Validation loop
        loss_sum = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataset):
            if validation_steps and batch_idx >= validation_steps:
                break

            # Execute validation step
            step_results = self.val_step(batch)
            loss_val = float(step_results['val_loss'])
            loss_sum += loss_val
            num_batches += 1

        # Compute validation metrics
        avg_loss = loss_sum / max(num_batches, 1)
        metrics_dict = {'val_loss': avg_loss}

        for metric in self.metrics:
            metrics_dict[f'val_{metric.name}'] = float(metric.result())

        return metrics_dict

    def save_model(self, filepath: str) -> None:
        """
        Save model to file.

        Args:
            filepath: Path to save model (should end with .keras).
        """
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")

    def save_history(self, filepath: str) -> None:
        """
        Save training history to JSON file.

        Args:
            filepath: Path to save history JSON.
        """
        with open(filepath, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        logger.info(f"Training history saved to {filepath}")


# ---------------------------------------------------------------------
# Inference Utilities
# ---------------------------------------------------------------------


class CLIPInference:
    """
    Enhanced inference utilities for trained CLIP models.

    Provides methods for:
    - Encoding images and texts to embeddings
    - Computing similarities
    - Zero-shot retrieval
    - Batch processing for efficiency

    Args:
        model: Trained CLIP model instance.
        data_loader: Data loader instance for preprocessing.
        batch_size: Batch size for inference (larger = faster).

    Example:
        ```python
        # Load trained model
        model = keras.models.load_model('clip_model.keras')

        # Create inference engine
        loader = CLIPDataLoader(image_size=224, context_length=77)
        inference = CLIPInference(model, loader, batch_size=64)

        # Encode images
        image_features = inference.encode_images(image_paths)

        # Encode texts
        text_features = inference.encode_texts(captions)

        # Compute similarities
        similarities = inference.compute_similarity(image_features, text_features)

        # Retrieve top matches
        results = inference.retrieve_images(
            query_text="a cat on a mat",
            image_paths=candidate_images,
            top_k=10
        )
        ```
    """

    def __init__(
        self,
        model: Any,  # CLIP model
        data_loader: CLIPDataLoader,
        batch_size: int = 64
    ) -> None:
        """Initialize CLIP inference engine."""
        self.model = model
        self.data_loader = data_loader
        self.batch_size = batch_size

        logger.info(
            f"CLIPInference initialized with batch_size={batch_size}"
        )

    def encode_images(
        self,
        image_paths: List[str],
        normalize: bool = True,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode batch of images to embeddings.

        Args:
            image_paths: List of image file paths.
            normalize: Whether to L2-normalize embeddings.
            show_progress: Whether to show progress bar.

        Returns:
            Array of image embeddings, shape (num_images, embed_dim).
        """
        num_images = len(image_paths)
        embeddings_list = []

        logger.info(f"Encoding {num_images} images...")

        # Process in batches
        for i in range(0, num_images, self.batch_size):
            batch_paths = image_paths[i:i + self.batch_size]

            # Load and preprocess images
            batch_images = []
            for path in batch_paths:
                try:
                    image = self.data_loader.preprocess_image(path, training=False)
                    batch_images.append(image.numpy())
                except Exception as e:
                    logger.error(f"Error loading image {path}: {e}")
                    # Use zero tensor as placeholder
                    batch_images.append(
                        np.zeros((self.data_loader.image_size,
                                self.data_loader.image_size, 3))
                    )

            # Convert to tensor
            batch_tensor = ops.convert_to_tensor(np.array(batch_images))

            # Encode
            batch_embeddings = self.model.encode_image(
                batch_tensor, training=False
            )
            embeddings_list.append(ops.convert_to_numpy(batch_embeddings))

            if show_progress and (i // self.batch_size) % 10 == 0:
                logger.info(f"  Processed {i + len(batch_paths)}/{num_images} images")

        # Concatenate all batches
        embeddings = np.vstack(embeddings_list)

        # Normalize if requested
        if normalize:
            embeddings = embeddings / np.linalg.norm(
                embeddings, axis=1, keepdims=True
            )

        logger.info(f"Encoded {num_images} images to shape {embeddings.shape}")
        return embeddings

    def encode_texts(
        self,
        texts: List[str],
        normalize: bool = True,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode batch of texts to embeddings.

        Args:
            texts: List of text strings.
            normalize: Whether to L2-normalize embeddings.
            show_progress: Whether to show progress bar.

        Returns:
            Array of text embeddings, shape (num_texts, embed_dim).
        """
        num_texts = len(texts)
        embeddings_list = []

        logger.info(f"Encoding {num_texts} texts...")

        # Process in batches
        for i in range(0, num_texts, self.batch_size):
            batch_texts = texts[i:i + self.batch_size]

            # Tokenize texts
            batch_tokens = []
            for text in batch_texts:
                try:
                    tokens = self.data_loader.preprocess_text(text)
                    batch_tokens.append(tokens.numpy())
                except Exception as e:
                    logger.error(f"Error tokenizing text '{text[:50]}...': {e}")
                    # Use zero tokens as placeholder
                    batch_tokens.append(
                        np.zeros(self.data_loader.context_length, dtype=np.int32)
                    )

            # Convert to tensor
            batch_tensor = ops.convert_to_tensor(np.array(batch_tokens))

            # Encode
            batch_embeddings = self.model.encode_text(
                batch_tensor, training=False
            )
            embeddings_list.append(ops.convert_to_numpy(batch_embeddings))

            if show_progress and (i // self.batch_size) % 10 == 0:
                logger.info(f"  Processed {i + len(batch_texts)}/{num_texts} texts")

        # Concatenate all batches
        embeddings = np.vstack(embeddings_list)

        # Normalize if requested
        if normalize:
            embeddings = embeddings / np.linalg.norm(
                embeddings, axis=1, keepdims=True
            )

        logger.info(f"Encoded {num_texts} texts to shape {embeddings.shape}")
        return embeddings

    def compute_similarity(
        self,
        image_embeddings: np.ndarray,
        text_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between image and text embeddings.

        Args:
            image_embeddings: Image embeddings, shape (num_images, embed_dim).
            text_embeddings: Text embeddings, shape (num_texts, embed_dim).

        Returns:
            Similarity matrix, shape (num_images, num_texts).
        """
        # Ensure embeddings are normalized
        image_embeddings = image_embeddings / np.linalg.norm(
            image_embeddings, axis=1, keepdims=True
        )
        text_embeddings = text_embeddings / np.linalg.norm(
            text_embeddings, axis=1, keepdims=True
        )

        # Compute dot product (cosine similarity for normalized vectors)
        similarities = np.dot(image_embeddings, text_embeddings.T)

        return similarities

    def retrieve_images(
        self,
        query_text: str,
        image_paths: List[str],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Retrieve top-k images for a text query.

        Args:
            query_text: Query text string.
            image_paths: List of candidate image paths.
            top_k: Number of top results to return.

        Returns:
            List of (image_path, similarity_score) tuples, sorted by score.
        """
        logger.info(f"Retrieving top-{top_k} images for query: '{query_text}'")

        # Encode query text
        text_embedding = self.encode_texts([query_text], show_progress=False)

        # Encode candidate images
        image_embeddings = self.encode_images(image_paths, show_progress=True)

        # Compute similarities
        similarities = self.compute_similarity(image_embeddings, text_embedding)
        scores = similarities[:, 0]  # Get scores for single query

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Prepare results
        results = [
            (image_paths[i], float(scores[i]))
            for i in top_indices
        ]

        logger.info(f"Retrieved top-{top_k} images:")
        for i, (path, score) in enumerate(results, 1):
            logger.info(f"  {i}. {Path(path).name}: {score:.4f}")

        return results

    def retrieve_texts(
        self,
        query_image_path: str,
        texts: List[str],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Retrieve top-k texts for an image query.

        Args:
            query_image_path: Path to query image.
            texts: List of candidate text strings.
            top_k: Number of top results to return.

        Returns:
            List of (text, similarity_score) tuples, sorted by score.
        """
        logger.info(
            f"Retrieving top-{top_k} texts for image: "
            f"{Path(query_image_path).name}"
        )

        # Encode query image
        image_embedding = self.encode_images(
            [query_image_path], show_progress=False
        )

        # Encode candidate texts
        text_embeddings = self.encode_texts(texts, show_progress=True)

        # Compute similarities
        similarities = self.compute_similarity(image_embedding, text_embeddings)
        scores = similarities[0, :]  # Get scores for single query

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Prepare results
        results = [
            (texts[i], float(scores[i]))
            for i in top_indices
        ]

        logger.info(f"Retrieved top-{top_k} texts:")
        for i, (text, score) in enumerate(results, 1):
            text_preview = text[:50] + "..." if len(text) > 50 else text
            logger.info(f"  {i}. '{text_preview}': {score:.4f}")

        return results


# ---------------------------------------------------------------------