"""
BERT Fine-tuning Script for Sentiment Analysis
===============================================

A complete training script for fine-tuning a pre-trained BERT encoder on a
downstream sentiment analysis task.

This script demonstrates the full fine-tuning workflow:
1. Loading a pre-trained BERT encoder.
2. Loading and preprocessing the IMDB reviews dataset with labels.
3. Integrating a task-specific head (Sequence Classification) with the encoder.
4. Configuring and running the fine-tuning process with in-training analysis.
5. Saving the fine-tuned model for inference.
6. Evaluating the model on sample texts.

Requirements:
------------
    - tensorflow >= 2.18.0
    - keras >= 3.8.0
    - tensorflow-datasets
    - transformers (HuggingFace)

"""

import os
import keras
import tensorflow as tf
from typing import Optional, Tuple
import tensorflow_datasets as tfds
from transformers import BertTokenizer


# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.models.bert import BERT
from dl_techniques.utils.logger import logger
from dl_techniques.layers.nlp_heads import create_nlp_head, NLPTaskConfig
from dl_techniques.callbacks.analyzer_callback import EpochAnalyzerCallback
from dl_techniques.layers.nlp_heads.task_types import NLPTaskType


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


class FinetuneConfig:
    """Configuration for BERT Sentiment Analysis fine-tuning.

    :param pretrained_encoder_path: Path to the .keras file of the pretrained BERT encoder.
    :type pretrained_encoder_path: str
    :param num_classes: Number of output classes for the classification task.
    :type num_classes: int
    :param max_seq_length: Maximum sequence length for training and evaluation.
    :type max_seq_length: int
    :param batch_size: Training and validation batch size.
    :type batch_size: int
    :param num_epochs: Number of fine-tuning epochs.
    :type num_epochs: int
    :param learning_rate: Peak learning rate for the fine-tuning process.
    :type learning_rate: float
    :param weight_decay: Weight decay for AdamW optimizer.
    :type weight_decay: float
    :param save_dir: Directory to save the final fine-tuned model.
    :type save_dir: str
    :param log_dir: Directory for TensorBoard logs.
    :type log_dir: str
    :param max_samples: Maximum number of training samples (None for all).
    :type max_samples: Optional[int]
    """

    # Model and paths
    pretrained_encoder_path: str = "results/bert_mlm_pretrain/pretrained_bert_encoder_best.keras"
    save_dir: str = "results/bert_sentiment_finetune"
    log_dir: str = "results/bert_sentiment_finetune/logs"
    checkpoint_dir: str = "results/bert_sentiment_finetune/checkpoints"
    analysis_dir: str = "results/bert_sentiment_finetune/epoch_analysis"

    # Task configuration
    num_classes: int = 2  # IMDB: 0 (negative), 1 (positive)

    # Training configuration
    max_seq_length: int = 256
    batch_size: int = 16
    num_epochs: int = 5
    learning_rate: float = 3e-5
    weight_decay: float = 0.01

    # Data configuration
    dataset_name: str = "imdb_reviews"
    max_samples: Optional[int] = None

    # In-Training Analysis Configuration
    run_epoch_analysis: bool = True
    analysis_start_epoch: int = 1
    analysis_epoch_frequency: int = 1  # Analyze every epoch during short fine-tuning


# ---------------------------------------------------------------------
# Data Pipeline
# ---------------------------------------------------------------------


def create_tokenizer() -> BertTokenizer:
    """Create and configure BERT tokenizer."""
    logger.info("Loading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer


def load_dataset(config: FinetuneConfig, split: str = "train") -> tf.data.Dataset:
    """Load text dataset with labels from tensorflow-datasets."""
    logger.info(f"Loading {config.dataset_name} dataset ({split} split)...")
    dataset, _ = tfds.load(
        config.dataset_name,
        split=split,
        as_supervised=True,
        shuffle_files=True,
        with_info=True,
    )
    if config.max_samples is not None and "train" in split:
        dataset = dataset.take(config.max_samples)
        logger.info(f"Limited training data to {config.max_samples} samples")
    return dataset


def preprocess_dataset(
    dataset: tf.data.Dataset,
    tokenizer: BertTokenizer,
    config: FinetuneConfig
) -> tf.data.Dataset:
    """Preprocess text dataset for sentiment analysis fine-tuning."""
    logger.info("Preprocessing dataset for fine-tuning...")

    def tokenize_function(text: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Tokenize text and return flat tensors."""
        text = tf.compat.as_str_any(text)
        encoded = tokenizer(
            text,
            max_length=config.max_seq_length,
            truncation=True,
            padding='max_length',
            return_tensors='np'
        )
        return (
            encoded['input_ids'][0],
            encoded['attention_mask'][0],
            encoded['token_type_ids'][0],
            label
        )

    dataset = dataset.map(
        lambda text, label: tf.py_function(
            func=tokenize_function,
            inp=[text, label],
            Tout=[tf.int32, tf.int32, tf.int32, tf.int64]
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    def structure_to_dict_and_label(input_ids, attention_mask, token_type_ids, label):
        """Restructures flat tensors into the model's expected input format."""
        inputs = {
            'input_ids': tf.ensure_shape(input_ids, [config.max_seq_length]),
            'attention_mask': tf.ensure_shape(attention_mask, [config.max_seq_length]),
            'token_type_ids': tf.ensure_shape(token_type_ids, [config.max_seq_length]),
        }
        label = tf.cast(label, dtype=tf.int32)
        label.set_shape(())  # Set shape for label to ensure it's a scalar
        return inputs, label

    dataset = dataset.map(structure_to_dict_and_label, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(config.batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    logger.info(f"Dataset preprocessed: batch_size={config.batch_size}")
    return dataset


# ---------------------------------------------------------------------
# Model Creation
# ---------------------------------------------------------------------


def create_sentiment_model(config: FinetuneConfig) -> keras.Model:
    """Create a sentiment analysis model by combining a pretrained BERT with a task head."""
    logger.info("=" * 80)
    logger.info("Creating Sentiment Analysis Model")
    logger.info("=" * 80)

    logger.info(f"Loading pretrained encoder from: {config.pretrained_encoder_path}")
    try:
        bert_encoder = keras.models.load_model(config.pretrained_encoder_path)
        logger.info("Pretrained BERT encoder loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load BERT encoder. Ensure the path is correct. Error: {e}")
        raise

    sentiment_task_config = NLPTaskConfig(
        name="sentiment_analysis",
        task_type=NLPTaskType.SENTIMENT_ANALYSIS,
        num_classes=config.num_classes,
        dropout_rate=0.1
    )
    logger.info(f"Task config: {sentiment_task_config.name}, num_classes={config.num_classes}")

    classification_head = create_nlp_head(
        task_config=sentiment_task_config,
        input_dim=bert_encoder.hidden_size
    )
    logger.info("Sequence classification head created.")

    inputs = {
        "input_ids": keras.Input(shape=(config.max_seq_length,), dtype="int32", name="input_ids"),
        "attention_mask": keras.Input(shape=(config.max_seq_length,), dtype="int32", name="attention_mask"),
        "token_type_ids": keras.Input(shape=(config.max_seq_length,), dtype="int32", name="token_type_ids"),
    }

    encoder_outputs = bert_encoder(inputs)
    sequence_output = encoder_outputs["last_hidden_state"]
    cls_token_output = sequence_output[:, 0, :]
    head_inputs = {"hidden_states": cls_token_output}
    task_outputs = classification_head(head_inputs)

    model = keras.Model(
        inputs=inputs,
        outputs=task_outputs['logits'],
        name="bert_sentiment_analyzer"
    )

    logger.info(f"Model created successfully. Total parameters: {model.count_params():,}")
    model.summary(print_fn=logger.info)

    return model


# ---------------------------------------------------------------------
# Training Configuration
# ---------------------------------------------------------------------


def compile_model(model: keras.Model, config: FinetuneConfig) -> None:
    """Compile the model with optimizer, loss, and metrics for fine-tuning."""
    logger.info("Compiling model for fine-tuning...")
    optimizer = keras.optimizers.AdamW(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay
    )
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [keras.metrics.SparseCategoricalAccuracy("accuracy")]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    logger.info("Model compiled.")


def create_callbacks(config: FinetuneConfig) -> list:
    """Create training callbacks, including the EpochAnalyzer."""
    logger.info("Creating training callbacks...")
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    best_model_filepath = os.path.join(config.checkpoint_dir, "best_sentiment_model.keras")

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=best_model_filepath,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.TensorBoard(log_dir=config.log_dir),
        keras.callbacks.CSVLogger(os.path.join(config.save_dir, "finetuning_log.csv")),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    # Conditionally add the EpochAnalyzerCallback, mirroring the pre-training script
    if config.run_epoch_analysis:
        analysis_callback = EpochAnalyzerCallback(
            output_dir=config.analysis_dir,
            start_epoch=config.analysis_start_epoch,
            epoch_frequency=config.analysis_epoch_frequency,
            model_name="BERT-Sentiment-Finetuned"
        )
        callbacks.append(analysis_callback)

    logger.info(f"Created {len(callbacks)} callbacks.")
    return callbacks


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------


def finetune_sentiment_model(config: FinetuneConfig) -> Tuple[keras.Model, keras.callbacks.History]:
    """Main function for fine-tuning the sentiment analysis model."""
    logger.info("=" * 80)
    logger.info("BERT Sentiment Analysis Fine-tuning")
    logger.info("=" * 80)

    tf.random.set_seed(42)
    keras.utils.set_random_seed(42)
    os.makedirs(config.save_dir, exist_ok=True)

    tokenizer = create_tokenizer()
    train_dataset = preprocess_dataset(load_dataset(config, split="train"), tokenizer, config)
    val_dataset = preprocess_dataset(load_dataset(config, split="test"), tokenizer, config)
    model = create_sentiment_model(config)
    compile_model(model, config)
    callbacks = create_callbacks(config)

    logger.info("=" * 80)
    logger.info("Starting fine-tuning...")
    logger.info("=" * 80)

    history = model.fit(
        train_dataset,
        epochs=config.num_epochs,
        callbacks=callbacks,
        validation_data=val_dataset,
        verbose=1,
    )

    logger.info("=" * 80)
    logger.info("Fine-tuning completed!")
    logger.info("=" * 80)

    final_model_path = os.path.join(config.save_dir, "bert_sentiment_final_best.keras")
    logger.info(f"Saving final fine-tuned model to {final_model_path}")
    model.save(final_model_path)

    best_epoch = tf.argmax(history.history['val_accuracy']).numpy()
    best_val_acc = history.history['val_accuracy'][best_epoch]
    logger.info(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch + 1}")

    return model, history


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------


def evaluate_model(model: keras.Model, tokenizer: BertTokenizer, config: FinetuneConfig) -> None:
    """Evaluate the fine-tuned model on sample sentences."""
    logger.info("=" * 80)
    logger.info("Model Evaluation on Sample Texts")
    logger.info("=" * 80)

    test_texts = [
        "This movie was an absolute masterpiece. The acting, story, and cinematography were all perfect.",
        "I was really disappointed with this film. It was boring and the plot made no sense.",
        "The movie was okay, not great but not terrible either. It's a decent way to spend an evening.",
        "A truly inspiring and heartwarming story that will stay with you long after you watch it.",
    ]
    labels = ["Negative", "Positive"]

    for text in test_texts:
        inputs = tokenizer(
            text,
            max_length=config.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="tf",
        )
        logits = model.predict({
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'token_type_ids': inputs['token_type_ids'],
        })
        probabilities = tf.nn.softmax(logits, axis=-1)[0]
        predicted_class_id = tf.argmax(probabilities).numpy()
        predicted_label = labels[predicted_class_id]
        logger.info(f"Text: '{text}'")
        logger.info(f" -> Predicted: '{predicted_label}' (Confidence: {probabilities[predicted_class_id]:.4f})")
        logger.info("-" * 20)


# ---------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------


def main() -> None:
    """Main entry point for BERT fine-tuning script."""
    config = FinetuneConfig()

    logger.info("Fine-tuning Configuration:")
    for key, value in vars(config).items():
        if not key.startswith("__"):
            logger.info(f"  - {key}: {value}")

    model, _ = finetune_sentiment_model(config)
    tokenizer = create_tokenizer()
    evaluate_model(model, tokenizer, config)

    logger.info("=" * 80)
    logger.info("Fine-tuning complete! Model is ready for inference.")
    logger.info(f"Load model: keras.models.load_model('{config.save_dir}/bert_sentiment_final_best.keras')")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()