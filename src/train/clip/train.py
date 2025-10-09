"""
Modern CLIP Usage Example

This example demonstrates how to train and use the Modern CLIP model
with the dl_techniques framework, including:

1. Model creation and configuration
2. Data preparation and loading
3. Training with SigLIP loss
4. Inference and evaluation
5. Cross-modal retrieval
6. Integration with diffusion models
"""

import json
import keras
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Optional

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.models.clip.model import (
    ModernCLIP,
    CLIPConfig,
    create_modern_clip,
    create_modern_clip_small
)
from dl_techniques.losses.siglip_contrastive_loss import (
    create_siglip_loss,
)
from dl_techniques.optimization import (
    optimizer_builder,
    learning_rate_schedule_builder
)
from dl_techniques.utils.logger import logger
from dl_techniques.layers.tokenizers.bpe import BPETokenizer

# ---------------------------------------------------------------------

class CLIPDataLoader:
    """Data loader for CLIP training with image-text pairs."""

    def __init__(
            self,
            image_size: Tuple[int, int] = (224, 224),
            max_text_length: int = 77,
            tokenizer: Optional[BPETokenizer] = None
    ):
        self.image_size = image_size
        self.max_text_length = max_text_length
        self.tokenizer = tokenizer or self._create_simple_tokenizer()

    def _create_simple_tokenizer(self):
        """Create a simple tokenizer for demo purposes."""
        # In practice, you'd use a proper BPE tokenizer
        # For now, we'll simulate with a basic word-level tokenizer
        vocab = {f"word_{i}": i for i in range(1000)}
        vocab["<pad>"] = 0
        vocab["<unk>"] = 1
        return vocab

    def preprocess_image(self, image_path: str) -> tf.Tensor:
        """Preprocess image for CLIP."""
        # Load and decode image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3)
        image = tf.cast(image, tf.float32)

        # Resize and normalize
        image = tf.image.resize(image, self.image_size)
        image = image / 255.0

        # Normalize to [-1, 1] (common for modern models)
        image = (image - 0.5) * 2.0

        return image

    def preprocess_text(self, text: str) -> tf.Tensor:
        """Preprocess text for CLIP."""
        # Simple tokenization (replace with proper BPE in practice)
        words = text.lower().split()[:self.max_text_length - 2]  # Leave space for special tokens

        # Convert to token IDs (simplified)
        if isinstance(self.tokenizer, dict):
            token_ids = [self.tokenizer.get(word, 1) for word in words]  # 1 = <unk>
        else:
            # Assume it's a proper tokenizer
            token_ids = self.tokenizer.encode(text)[:self.max_text_length]

        # Pad to max length
        token_ids = token_ids + [0] * (self.max_text_length - len(token_ids))

        return tf.constant(token_ids[:self.max_text_length], dtype=tf.int32)

    def create_dataset(
            self,
            image_paths: List[str],
            captions: List[str],
            batch_size: int = 32,
            shuffle: bool = True
    ) -> tf.data.Dataset:
        """Create TensorFlow dataset for training."""
        assert len(image_paths) == len(captions), "Must have equal number of images and captions"

        # Create dataset from paths and captions
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, captions))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(image_paths))

        # Map preprocessing functions
        dataset = dataset.map(
            lambda img_path, caption: (
                self.preprocess_image(img_path),
                self.preprocess_text(caption)
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Batch and prefetch
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset


class CLIPTrainer:
    """Trainer class for Modern CLIP models."""

    def __init__(
            self,
            model: ModernCLIP,
            loss_fn: keras.losses.Loss,
            optimizer: keras.optimizers.Optimizer,
            metrics: Optional[List[keras.metrics.Metric]] = None
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metrics = metrics or []

        # Training state
        self.current_epoch = 0
        self.training_history = {
            'loss': [],
            'val_loss': [],
            'learning_rate': []
        }

    @tf.function
    def train_step(self, batch):
        """Single training step."""
        images, texts = batch

        with tf.GradientTape() as tape:
            # Forward pass
            outputs = self.model([images, texts], training=True)

            # Compute loss (SigLIP doesn't need labels)
            loss = self.loss_fn(None, outputs)

            # Add regularization losses
            if self.model.losses:
                loss += tf.add_n(self.model.losses)

        # Backward pass
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Update metrics
        results = {'loss': loss}
        for metric in self.metrics:
            metric.update_state(None, outputs)
            results[metric.name] = metric.result()

        return results

    @tf.function
    def val_step(self, batch):
        """Single validation step."""
        images, texts = batch

        # Forward pass
        outputs = self.model([images, texts], training=False)

        # Compute loss
        loss = self.loss_fn(None, outputs)

        # Update metrics
        results = {'val_loss': loss}
        for metric in self.metrics:
            metric.update_state(None, outputs)
            results[f'val_{metric.name}'] = metric.result()

        return results

    def fit(
            self,
            train_dataset: tf.data.Dataset,
            validation_dataset: Optional[tf.data.Dataset] = None,
            epochs: int = 10,
            steps_per_epoch: Optional[int] = None,
            validation_steps: Optional[int] = None,
            callbacks: Optional[List[keras.callbacks.Callback]] = None,
            verbose: int = 1
    ):
        """Train the model."""
        logger.info(f"Starting training for {epochs} epochs")

        callbacks = callbacks or []

        for epoch in range(epochs):
            self.current_epoch = epoch

            # Training phase
            if verbose >= 1:
                logger.info(f"Epoch {epoch + 1}/{epochs}")

            # Reset metrics
            for metric in self.metrics:
                metric.reset_state()

            # Training loop
            train_loss = 0
            num_batches = 0

            for batch_idx, batch in enumerate(train_dataset):
                if steps_per_epoch and batch_idx >= steps_per_epoch:
                    break

                results = self.train_step(batch)
                train_loss += results['loss']
                num_batches += 1

                if verbose >= 2 and batch_idx % 100 == 0:
                    current_lr = self.optimizer.learning_rate
                    if hasattr(current_lr, 'numpy'):
                        current_lr = current_lr.numpy()
                    logger.info(
                        f"  Batch {batch_idx}: loss={results['loss']:.4f}, lr={current_lr:.6f}"
                    )

            avg_train_loss = train_loss / num_batches
            self.training_history['loss'].append(float(avg_train_loss))

            # Validation phase
            if validation_dataset:
                val_loss = 0
                val_batches = 0

                for batch_idx, batch in enumerate(validation_dataset):
                    if validation_steps and batch_idx >= validation_steps:
                        break

                    results = self.val_step(batch)
                    val_loss += results['val_loss']
                    val_batches += 1

                avg_val_loss = val_loss / val_batches
                self.training_history['val_loss'].append(float(avg_val_loss))

                if verbose >= 1:
                    logger.info(
                        f"  train_loss: {avg_train_loss:.4f} - val_loss: {avg_val_loss:.4f}"
                    )
            else:
                if verbose >= 1:
                    logger.info(f"  train_loss: {avg_train_loss:.4f}")

            # Learning rate tracking
            current_lr = self.optimizer.learning_rate
            if hasattr(current_lr, 'numpy'):
                current_lr = current_lr.numpy()
            self.training_history['learning_rate'].append(float(current_lr))

            # Execute callbacks
            for callback in callbacks:
                if hasattr(callback, 'on_epoch_end'):
                    callback.on_epoch_end(epoch, {
                        'loss': avg_train_loss,
                        'val_loss': avg_val_loss if validation_dataset else None
                    })

    def save_model(self, filepath: str):
        """Save the trained model."""
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")

    def save_history(self, filepath: str):
        """Save training history."""
        with open(filepath, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        logger.info(f"Training history saved to {filepath}")


class CLIPInference:
    """Inference utilities for trained CLIP models."""

    def __init__(self, model: ModernCLIP, data_loader: CLIPDataLoader):
        self.model = model
        self.data_loader = data_loader

    def encode_images(self, image_paths: List[str]) -> np.ndarray:
        """Encode list of images to embeddings."""
        embeddings = []

        for image_path in image_paths:
            image = self.data_loader.preprocess_image(image_path)
            image = tf.expand_dims(image, 0)  # Add batch dimension

            embedding = self.model.encode_image(image, training=False)
            embeddings.append(embedding.numpy())

        return np.vstack(embeddings)

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode list of texts to embeddings."""
        embeddings = []

        for text in texts:
            tokens = self.data_loader.preprocess_text(text)
            tokens = tf.expand_dims(tokens, 0)  # Add batch dimension

            embedding = self.model.encode_text(tokens, training=False)
            embeddings.append(embedding.numpy())

        return np.vstack(embeddings)

    def compute_similarity(self, image_embeddings: np.ndarray, text_embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between image and text embeddings."""
        # Normalize embeddings
        image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
        text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)

        # Compute similarity matrix
        similarity = np.dot(image_embeddings, text_embeddings.T)
        return similarity

    def retrieve_images(
            self,
            query_text: str,
            image_paths: List[str],
            top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Retrieve most similar images for a text query."""
        # Encode query
        text_embedding = self.encode_texts([query_text])

        # Encode images
        image_embeddings = self.encode_images(image_paths)

        # Compute similarities
        similarities = self.compute_similarity(image_embeddings, text_embedding)
        scores = similarities[:, 0]  # Single query

        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(image_paths[i], float(scores[i])) for i in top_indices]

        return results

    def retrieve_texts(
            self,
            query_image_path: str,
            texts: List[str],
            top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Retrieve most similar texts for an image query."""
        # Encode query
        image_embedding = self.encode_images([query_image_path])

        # Encode texts
        text_embeddings = self.encode_texts(texts)

        # Compute similarities
        similarities = self.compute_similarity(image_embedding, text_embeddings)
        scores = similarities[0, :]  # Single query

        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(texts[i], float(scores[i])) for i in top_indices]

        return results


def main_training_example():
    """Example of training a Modern CLIP model."""
    logger.info("=== Modern CLIP Training Example ===")

    # 1. Create model
    config = CLIPConfig.get_variant_config("small")
    config.use_flash_attention = True  # Enable Flash Attention if available
    config.learnable_temperature = True

    model = ModernCLIP(config)
    logger.info(f"Created {config.variant} CLIP model with {model.count_params():,} parameters")

    # 2. Create loss function
    loss_fn = create_siglip_loss(
        temperature=0.07,
        use_learnable_temperature=True
    )

    # 3. Create optimizer with learning rate schedule
    lr_config = {
        "type": "cosine_decay",
        "warmup_steps": 2000,
        "warmup_start_lr": 1e-8,
        "learning_rate": 3e-4,
        "decay_steps": 50000,
        "alpha": 0.01
    }

    opt_config = {
        "type": "adamw",
        "beta_1": 0.9,
        "beta_2": 0.98,
        "gradient_clipping_by_norm": 1.0
    }

    lr_schedule = learning_rate_schedule_builder(lr_config)
    optimizer = optimizer_builder(opt_config, lr_schedule)

    # 4. Create data loader (dummy data for example)
    data_loader = CLIPDataLoader(
        image_size=config.image_size,
        max_text_length=config.max_text_length
    )

    # Create dummy dataset
    dummy_images = tf.random.normal((100, *config.image_size, 3))
    dummy_texts = tf.random.randint(1, 1000, (100, config.max_text_length))

    train_dataset = tf.data.Dataset.from_tensor_slices((dummy_images, dummy_texts))
    train_dataset = train_dataset.batch(8).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((dummy_images[:20], dummy_texts[:20]))
    val_dataset = val_dataset.batch(8).prefetch(tf.data.AUTOTUNE)

    # 5. Create trainer and train
    trainer = CLIPTrainer(model, loss_fn, optimizer)

    # Training callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7
        )
    ]

    # Train model
    trainer.fit(
        train_dataset=train_dataset,
        validation_dataset=val_dataset,
        epochs=10,
        steps_per_epoch=10,  # Small for demo
        validation_steps=3,
        callbacks=callbacks,
        verbose=1
    )

    # 6. Save model
    model.save("modern_clip_small.keras")
    trainer.save_history("training_history.json")

    logger.info("Training completed!")


def main_inference_example():
    """Example of using trained CLIP model for inference."""
    logger.info("=== Modern CLIP Inference Example ===")

    # Load trained model (or create new for demo)
    model = create_modern_clip_small()

    # Create data loader
    data_loader = CLIPDataLoader()

    # Create inference utility
    inference = CLIPInference(model, data_loader)

    # Example texts and image paths (dummy for demonstration)
    texts = [
        "A cat sitting on a windowsill",
        "A dog playing in the park",
        "A beautiful sunset over mountains",
        "A busy city street at night",
        "A child reading a book"
    ]

    image_paths = [f"image_{i}.jpg" for i in range(10)]  # Dummy paths

    # Simulate encoding (in practice, you'd have real images)
    logger.info("Encoding texts...")
    text_embeddings = np.random.random((len(texts), model.config.projection_dim))
    text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)

    logger.info("Encoding images...")
    image_embeddings = np.random.random((len(image_paths), model.config.projection_dim))
    image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)

    # Compute similarity matrix
    similarity_matrix = np.dot(image_embeddings, text_embeddings.T)

    logger.info("Image-Text Similarity Matrix:")
    logger.info(f"Shape: {similarity_matrix.shape}")
    logger.info(f"Max similarity: {similarity_matrix.max():.3f}")
    logger.info(f"Min similarity: {similarity_matrix.min():.3f}")

    # Find best matches
    for i, text in enumerate(texts):
        best_image_idx = np.argmax(similarity_matrix[:, i])
        best_score = similarity_matrix[best_image_idx, i]
        logger.info(f"Text: '{text}' -> Best image: {image_paths[best_image_idx]} (score: {best_score:.3f})")


def main_cross_modal_retrieval_example():
    """Example of cross-modal retrieval with CLIP."""
    logger.info("=== Cross-Modal Retrieval Example ===")

    # Create small model for demo
    model = create_modern_clip_small()
    data_loader = CLIPDataLoader()
    inference = CLIPInference(model, data_loader)

    # Simulate retrieval scenarios
    query_texts = [
        "a red sports car",
        "a fluffy white cat",
        "sunset over ocean"
    ]

    image_database = [f"img_{i:03d}.jpg" for i in range(100)]

    logger.info("Performing text-to-image retrieval...")
    for query in query_texts:
        # In practice, this would use real images and compute actual similarities
        # For demo, we'll simulate the process
        logger.info(f"Query: '{query}'")
        logger.info("  Top matches: img_042.jpg (0.89), img_015.jpg (0.84), img_078.jpg (0.81)")

    logger.info("Cross-modal retrieval simulation completed!")


if __name__ == "__main__":
    # Run examples
    try:
        main_training_example()
        main_inference_example()
        main_cross_modal_retrieval_example()
    except Exception as e:
        logger.error(f"Error in examples: {e}")
        raise