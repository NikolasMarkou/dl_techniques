"""
Comprehensive training example and usage guide for Vision Language Models.

This module demonstrates how to train and use the VLM system for various
multimodal tasks including image captioning, VQA, and contrastive learning.
"""

import os
import keras
import numpy as np
from keras import ops
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.analyzer import ModelAnalyzer, AnalysisConfig, DataInput

from dl_techniques.layers.vision_language_heads import CompleteVLM, VLMTrainingUtils
from dl_techniques.models.vlm import (
    VisionLanguageModel,
    create_vlm_for_image_captioning,
    create_vlm_for_vqa,
    create_clip_style_vlm
)

# ---------------------------------------------------------------------

class VLMDataProcessor:
    """Data processor for Vision Language Models."""

    def __init__(
            self,
            image_size: Tuple[int, int] = (224, 224),
            max_text_length: int = 128,
            vocab_size: int = 50000
    ):
        self.image_size = image_size
        self.max_text_length = max_text_length
        self.vocab_size = vocab_size

    def preprocess_images(self, images: np.ndarray) -> np.ndarray:
        """
        Preprocess images for the model.

        Args:
            images: Raw images of shape (batch_size, height, width, channels).

        Returns:
            Preprocessed images.
        """
        # Resize to target size
        if images.shape[1:3] != self.image_size:
            images = keras.utils.image_utils.smart_resize(
                images, self.image_size, interpolation='bilinear'
            )

        # Normalize to [-1, 1]
        images = (images.astype(np.float32) / 127.5) - 1.0

        return images

    def preprocess_text(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess text for the model.

        Args:
            texts: List of text strings.

        Returns:
            Tuple of (token_ids, attention_mask).
        """
        # This is a simplified tokenization - in practice, use a proper tokenizer
        token_ids = np.zeros((len(texts), self.max_text_length), dtype=np.int32)
        attention_mask = np.zeros((len(texts), self.max_text_length), dtype=np.float32)

        for i, text in enumerate(texts):
            # Simple word-based tokenization (replace with proper tokenizer)
            words = text.lower().split()[:self.max_text_length - 2]  # Reserve space for special tokens

            # Add special tokens (1 = start, 2 = end, 0 = pad)
            tokens = [1] + [hash(word) % (self.vocab_size - 3) + 3 for word in words] + [2]
            seq_len = len(tokens)

            token_ids[i, :seq_len] = tokens
            attention_mask[i, :seq_len] = 1.0

        return token_ids, attention_mask

    def create_dummy_dataset(self, num_samples: int = 1000) -> Dict[str, np.ndarray]:
        """
        Create a dummy dataset for testing.

        Args:
            num_samples: Number of samples to generate.

        Returns:
            Dictionary containing dataset components.
        """
        logger.info(f"Creating dummy dataset with {num_samples} samples")

        # Generate random images
        images = np.random.uniform(0, 255, (num_samples, *self.image_size, 3)).astype(np.uint8)

        # Generate dummy captions
        dummy_words = ["cat", "dog", "person", "car", "tree", "house", "water", "sky", "grass", "flower"]
        captions = []
        for _ in range(num_samples):
            num_words = np.random.randint(5, 15)
            caption = " ".join(np.random.choice(dummy_words, num_words))
            captions.append(caption)

        # Generate dummy VQA pairs
        questions = []
        answers = []
        answer_classes = ["yes", "no", "red", "blue", "one", "two", "three", "cat", "dog", "person"]

        for _ in range(num_samples):
            question = f"What is in the image?"
            answer = np.random.choice(answer_classes)
            questions.append(question)
            answers.append(answer_classes.index(answer))

        # Preprocess data
        processed_images = self.preprocess_images(images)
        caption_tokens, caption_masks = self.preprocess_text(captions)
        question_tokens, question_masks = self.preprocess_text(questions)

        return {
            "images": processed_images,
            "captions": captions,
            "caption_tokens": caption_tokens,
            "caption_masks": caption_masks,
            "questions": questions,
            "question_tokens": question_tokens,
            "question_masks": question_masks,
            "answers": np.array(answers),
            "num_answer_classes": len(answer_classes)
        }

# ---------------------------------------------------------------------

class VLMTrainer:
    """Trainer class for Vision Language Models."""

    def __init__(
            self,
            model: keras.Model,
            optimizer_config: Dict[str, Any] = None,
            loss_weights: Dict[str, float] = None
    ):
        self.model = model
        self.optimizer_config = optimizer_config or {
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "warmup_steps": 1000,
            "total_steps": 10000
        }
        self.loss_weights = loss_weights or {"contrastive": 1.0, "captioning": 1.0, "vqa": 1.0}

        # Initialize optimizer
        self.optimizer = self._create_optimizer()

        # Metrics
        self.train_metrics = {
            "loss": keras.metrics.Mean(name="train_loss"),
            "contrastive_loss": keras.metrics.Mean(name="train_contrastive_loss"),
            "captioning_loss": keras.metrics.Mean(name="train_captioning_loss"),
            "vqa_accuracy": keras.metrics.CategoricalAccuracy(name="train_vqa_accuracy")
        }

        self.val_metrics = {
            "loss": keras.metrics.Mean(name="val_loss"),
            "contrastive_loss": keras.metrics.Mean(name="val_contrastive_loss"),
            "captioning_loss": keras.metrics.Mean(name="val_captioning_loss"),
            "vqa_accuracy": keras.metrics.CategoricalAccuracy(name="val_vqa_accuracy")
        }

    def _create_optimizer(self) -> keras.optimizers.Optimizer:
        """Create optimizer with warmup schedule."""
        # Create learning rate schedule
        lr_schedule = WarmupCosineDecay(
            learning_rate=self.optimizer_config["learning_rate"],
            warmup_steps=self.optimizer_config["warmup_steps"],
            total_steps=self.optimizer_config["total_steps"]
        )

        # Create optimizer
        optimizer = keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=self.optimizer_config["weight_decay"],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8
        )

        return optimizer

    @keras.utils.tf_utils.function
    def train_step(self, batch_data: Dict[str, keras.KerasTensor]) -> Dict[str, keras.KerasTensor]:
        """
        Single training step.

        Args:
            batch_data: Batch of training data.

        Returns:
            Dictionary of loss values and metrics.
        """
        with keras.utils.GradientTape() as tape:
            # Forward pass for different tasks
            contrastive_inputs = {
                "images": batch_data["images"],
                "text_tokens": batch_data["caption_tokens"],
                "attention_mask": batch_data["caption_masks"]
            }

            # Contrastive learning
            contrastive_outputs = self.model(contrastive_inputs, task="contrastive", training=True)
            contrastive_loss = VLMTrainingUtils.compute_contrastive_loss(
                contrastive_outputs["vision_projected"],
                contrastive_outputs["text_projected"]
            )

            # Image captioning
            captioning_outputs = self.model(contrastive_inputs, task="captioning", training=True)
            captioning_loss = VLMTrainingUtils.compute_captioning_loss(
                captioning_outputs,
                batch_data["caption_tokens"][:, 1:],  # Shift for next token prediction
                batch_data["caption_masks"][:, 1:]
            )

            # VQA
            vqa_inputs = {
                "images": batch_data["images"],
                "text_tokens": batch_data["question_tokens"],
                "attention_mask": batch_data["question_masks"]
            }
            vqa_outputs = self.model(vqa_inputs, task="vqa", training=True)
            vqa_loss = keras.losses.sparse_categorical_crossentropy(
                batch_data["answers"], vqa_outputs, from_logits=True
            )
            vqa_loss = ops.mean(vqa_loss)

            # Total loss
            total_loss = (
                    self.loss_weights["contrastive"] * contrastive_loss +
                    self.loss_weights["captioning"] * captioning_loss +
                    self.loss_weights["vqa"] * vqa_loss
            )

        # Compute gradients and update
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Update metrics
        self.train_metrics["loss"].update_state(total_loss)
        self.train_metrics["contrastive_loss"].update_state(contrastive_loss)
        self.train_metrics["captioning_loss"].update_state(captioning_loss)
        self.train_metrics["vqa_accuracy"].update_state(
            keras.utils.to_categorical(batch_data["answers"], batch_data.get("num_answer_classes", 10)),
            ops.softmax(vqa_outputs)
        )

        return {
            "total_loss": total_loss,
            "contrastive_loss": contrastive_loss,
            "captioning_loss": captioning_loss,
            "vqa_loss": vqa_loss
        }

    def validate_step(self, batch_data: Dict[str, keras.KerasTensor]) -> Dict[str, keras.KerasTensor]:
        """
        Single validation step.

        Args:
            batch_data: Batch of validation data.

        Returns:
            Dictionary of loss values and metrics.
        """
        # Similar to train_step but without gradient computation
        contrastive_inputs = {
            "images": batch_data["images"],
            "text_tokens": batch_data["caption_tokens"],
            "attention_mask": batch_data["caption_masks"]
        }

        # Forward passes
        contrastive_outputs = self.model(contrastive_inputs, task="contrastive", training=False)
        captioning_outputs = self.model(contrastive_inputs, task="captioning", training=False)

        vqa_inputs = {
            "images": batch_data["images"],
            "text_tokens": batch_data["question_tokens"],
            "attention_mask": batch_data["question_masks"]
        }
        vqa_outputs = self.model(vqa_inputs, task="vqa", training=False)

        # Compute losses
        contrastive_loss = VLMTrainingUtils.compute_contrastive_loss(
            contrastive_outputs["vision_projected"],
            contrastive_outputs["text_projected"]
        )

        captioning_loss = VLMTrainingUtils.compute_captioning_loss(
            captioning_outputs,
            batch_data["caption_tokens"][:, 1:],
            batch_data["caption_masks"][:, 1:]
        )

        vqa_loss = ops.mean(keras.losses.sparse_categorical_crossentropy(
            batch_data["answers"], vqa_outputs, from_logits=True
        ))

        total_loss = (
                self.loss_weights["contrastive"] * contrastive_loss +
                self.loss_weights["captioning"] * captioning_loss +
                self.loss_weights["vqa"] * vqa_loss
        )

        # Update validation metrics
        self.val_metrics["loss"].update_state(total_loss)
        self.val_metrics["contrastive_loss"].update_state(contrastive_loss)
        self.val_metrics["captioning_loss"].update_state(captioning_loss)
        self.val_metrics["vqa_accuracy"].update_state(
            keras.utils.to_categorical(batch_data["answers"], batch_data.get("num_answer_classes", 10)),
            ops.softmax(vqa_outputs)
        )

        return {
            "total_loss": total_loss,
            "contrastive_loss": contrastive_loss,
            "captioning_loss": captioning_loss,
            "vqa_loss": vqa_loss
        }

    def train(
            self,
            train_dataset: Dict[str, np.ndarray],
            val_dataset: Dict[str, np.ndarray],
            epochs: int = 10,
            batch_size: int = 32,
            save_path: str = "vlm_model.keras"
    ) -> Dict[str, List[float]]:
        """
        Train the VLM model.

        Args:
            train_dataset: Training dataset.
            val_dataset: Validation dataset.
            epochs: Number of training epochs.
            batch_size: Batch size.
            save_path: Path to save the trained model.

        Returns:
            Training history.
        """
        logger.info(f"Starting VLM training for {epochs} epochs")

        # Training history
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_contrastive_loss": [],
            "val_contrastive_loss": [],
            "train_captioning_loss": [],
            "val_captioning_loss": [],
            "train_vqa_accuracy": [],
            "val_vqa_accuracy": []
        }

        # Training loop
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")

            # Reset metrics
            for metric in self.train_metrics.values():
                metric.reset_state()
            for metric in self.val_metrics.values():
                metric.reset_state()

            # Training phase
            num_train_batches = len(train_dataset["images"]) // batch_size
            for batch_idx in range(num_train_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size

                batch_data = {
                    key: value[start_idx:end_idx]
                    for key, value in train_dataset.items()
                    if isinstance(value, np.ndarray)
                }
                batch_data["num_answer_classes"] = train_dataset["num_answer_classes"]

                # Convert to tensors
                batch_tensors = {
                    key: ops.convert_to_tensor(value) if isinstance(value, np.ndarray) else value
                    for key, value in batch_data.items()
                }

                # Training step
                self.train_step(batch_tensors)

                if batch_idx % 100 == 0:
                    logger.info(
                        f"Batch {batch_idx}/{num_train_batches}, "
                        f"Loss: {self.train_metrics['loss'].result():.4f}"
                    )

            # Validation phase
            num_val_batches = len(val_dataset["images"]) // batch_size
            for batch_idx in range(num_val_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size

                batch_data = {
                    key: value[start_idx:end_idx]
                    for key, value in val_dataset.items()
                    if isinstance(value, np.ndarray)
                }
                batch_data["num_answer_classes"] = val_dataset["num_answer_classes"]

                # Convert to tensors
                batch_tensors = {
                    key: ops.convert_to_tensor(value) if isinstance(value, np.ndarray) else value
                    for key, value in batch_data.items()
                }

                # Validation step
                self.validate_step(batch_tensors)

            # Log epoch results
            train_loss = float(self.train_metrics["loss"].result())
            val_loss = float(self.val_metrics["loss"].result())
            train_vqa_acc = float(self.train_metrics["vqa_accuracy"].result())
            val_vqa_acc = float(self.val_metrics["vqa_accuracy"].result())

            logger.info(
                f"Epoch {epoch + 1} Results:\n"
                f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n"
                f"  Train VQA Acc: {train_vqa_acc:.4f}, Val VQA Acc: {val_vqa_acc:.4f}"
            )

            # Update history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_contrastive_loss"].append(float(self.train_metrics["contrastive_loss"].result()))
            history["val_contrastive_loss"].append(float(self.val_metrics["contrastive_loss"].result()))
            history["train_captioning_loss"].append(float(self.train_metrics["captioning_loss"].result()))
            history["val_captioning_loss"].append(float(self.val_metrics["captioning_loss"].result()))
            history["train_vqa_accuracy"].append(train_vqa_acc)
            history["val_vqa_accuracy"].append(val_vqa_acc)

        # Save model
        self.model.save(save_path)
        logger.info(f"Model saved to {save_path}")

        return history

# ---------------------------------------------------------------------

def plot_training_history(history: Dict[str, List[float]], save_path: str = "training_history.png"):
    """
    Plot training history.

    Args:
        history: Training history dictionary.
        save_path: Path to save the plot.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss plots
    axes[0, 0].plot(history["train_loss"], label="Train Loss")
    axes[0, 0].plot(history["val_loss"], label="Val Loss")
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()

    # Contrastive loss
    axes[0, 1].plot(history["train_contrastive_loss"], label="Train Contrastive")
    axes[0, 1].plot(history["val_contrastive_loss"], label="Val Contrastive")
    axes[0, 1].set_title("Contrastive Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()

    # Captioning loss
    axes[1, 0].plot(history["train_captioning_loss"], label="Train Captioning")
    axes[1, 0].plot(history["val_captioning_loss"], label="Val Captioning")
    axes[1, 0].set_title("Captioning Loss")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend()

    # VQA accuracy
    axes[1, 1].plot(history["train_vqa_accuracy"], label="Train VQA Acc")
    axes[1, 1].plot(history["val_vqa_accuracy"], label="Val VQA Acc")
    axes[1, 1].set_title("VQA Accuracy")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Training history plot saved to {save_path}")

# ---------------------------------------------------------------------

def analyze_trained_model(model_path: str, test_data: Dict[str, np.ndarray], output_dir: str = "vlm_analysis"):
    """
    Analyze a trained VLM model using the model analyzer.

    Args:
        model_path: Path to the trained model.
        test_data: Test data for analysis.
        output_dir: Output directory for analysis results.
    """
    logger.info("Starting model analysis")

    # Load the trained model
    model = keras.models.load_model(model_path)

    # Prepare test data for analyzer
    test_images = test_data["images"][:100]  # Sample for analysis
    test_tokens = test_data["caption_tokens"][:100]

    # Create data input for analyzer (simplified for base model analysis)
    analyzer_data = DataInput(
        x_data=test_images,
        y_data=test_tokens
    )

    # Configure analysis
    analysis_config = AnalysisConfig(
        analyze_weights=True,
        analyze_calibration=False,  # Skip calibration for multimodal model
        analyze_information_flow=True,
        analyze_training_dynamics=False,  # No training history provided
        save_plots=True,
        plot_style="publication"
    )

    # Initialize analyzer
    analyzer = ModelAnalyzer(
        models={"VLM": model.base_model if hasattr(model, 'base_model') else model},
        config=analysis_config,
        output_dir=output_dir
    )

    # Run analysis
    try:
        results = analyzer.analyze(data=analyzer_data)
        logger.info(f"Model analysis completed. Results saved to {output_dir}")

        # Print summary
        summary = analyzer.get_summary_statistics()
        if "weight_summary" in summary:
            for model_name, stats in summary["weight_summary"].items():
                logger.info(f"{model_name} - Total Parameters: {stats.get('total_parameters', 'N/A')}")

    except Exception as e:
        logger.warning(f"Model analysis failed: {e}")

# ---------------------------------------------------------------------

def main():
    """Main training and evaluation pipeline."""
    logger.info("Starting VLM training pipeline")

    # Create data processor
    data_processor = VLMDataProcessor(
        image_size=(224, 224),
        max_text_length=128,
        vocab_size=50000
    )

    # Create dummy dataset
    full_dataset = data_processor.create_dummy_dataset(num_samples=2000)

    # Split dataset
    split_idx = int(0.8 * len(full_dataset["images"]))
    train_dataset = {key: value[:split_idx] if isinstance(value, np.ndarray) else value
                     for key, value in full_dataset.items()}
    val_dataset = {key: value[split_idx:] if isinstance(value, np.ndarray) else value
                   for key, value in full_dataset.items()}

    logger.info(f"Training samples: {len(train_dataset['images'])}")
    logger.info(f"Validation samples: {len(val_dataset['images'])}")

    # Create base VLM model
    from vlm_model import create_vlm_for_image_captioning
    base_vlm = create_vlm_for_image_captioning(
        image_size=(224, 224),
        vocab_size=50000,
        max_text_length=128
    )

    # Create complete VLM with task heads
    from vlm_task_heads import CompleteVLM
    task_configs = {
        "captioning": {
            "vocab_size": 50000,
            "embed_dim": 768,
            "num_layers": 4,  # Smaller for demo
            "num_heads": 12,
        },
        "vqa": {
            "num_answers": full_dataset["num_answer_classes"],
            "embed_dim": 768,
            "hidden_dims": [512, 256],
        },
        "contrastive": {
            "embed_dim": 768,
            "projection_dim": 256,
            "temperature": 0.07,
        }
    }

    complete_vlm = CompleteVLM(
        base_model=base_vlm,
        task_configs=task_configs
    )

    # Create trainer
    trainer = VLMTrainer(
        model=complete_vlm,
        optimizer_config={
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "warmup_steps": 100,
            "total_steps": 1000
        },
        loss_weights={"contrastive": 1.0, "captioning": 1.0, "vqa": 1.0}
    )

    # Train model
    history = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=3,  # Small number for demo
        batch_size=8,  # Small batch size for demo
        save_path="trained_vlm.keras"
    )

    # Plot training history
    plot_training_history(history, "vlm_training_history.png")

    # Analyze trained model
    analyze_trained_model("trained_vlm.keras", val_dataset, "vlm_analysis")

    logger.info("VLM training pipeline completed successfully!")

# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()