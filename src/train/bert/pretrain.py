"""BERT Pre-training Script with Masked Language Modeling.

Pre-trains a BERT encoder using MLM on a text dataset (IMDB reviews by default).
Saves both the full MLM model and the encoder separately for downstream fine-tuning.
"""

import argparse
import os

import keras
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Dict, List, Optional, Tuple

from train.common import setup_gpu

from dl_techniques.models.bert import BERT
from dl_techniques.models.masked_language_model import (
    MaskedLanguageModel,
    visualize_mlm_predictions,
)
from dl_techniques.utils.logger import logger
from dl_techniques.utils.tokenizer import TiktokenPreprocessor
from dl_techniques.optimization.warmup_schedule import WarmupSchedule
from dl_techniques.callbacks.analyzer_callback import EpochAnalyzerCallback


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


class TrainingConfig:
    """Configuration for BERT MLM pre-training."""

    # Model
    bert_variant: str = "tiny"
    vocab_size: int = 100277  # Tiktoken cl100k_base
    max_seq_length: int = 128

    # Tokenizer (Tiktoken cl100k_base)
    encoding_name: str = "cl100k_base"
    cls_token_id: int = 100264
    sep_token_id: int = 100265
    pad_token_id: int = 100266
    mask_token_id: int = 100267

    # Training
    batch_size: int = 32
    num_epochs: int = 3
    learning_rate: float = 5e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

    # MLM
    mask_ratio: float = 0.15
    random_token_ratio: float = 0.1
    unchanged_ratio: float = 0.1

    # Paths
    save_dir: str = "results/bert_pretrain"
    log_dir: str = "results/bert_pretrain/logs"
    checkpoint_dir: str = "results/bert_pretrain/checkpoints"
    analysis_dir: str = "results/bert_pretrain/epoch_analysis"

    # Data
    dataset_name: str = "imdb_reviews"
    max_samples: Optional[int] = 10000

    # Analysis
    run_epoch_analysis: bool = True
    analysis_start_epoch: int = 1
    analysis_epoch_frequency: int = 5


# ---------------------------------------------------------------------
# Data Pipeline
# ---------------------------------------------------------------------


def create_tokenizer(config: TrainingConfig) -> TiktokenPreprocessor:
    """Create and configure Tiktoken preprocessor."""
    preprocessor = TiktokenPreprocessor(
        encoding_name=config.encoding_name,
        max_length=config.max_seq_length,
        cls_token_id=config.cls_token_id,
        sep_token_id=config.sep_token_id,
        pad_token_id=config.pad_token_id,
        mask_token_id=config.mask_token_id,
        truncation=True,
        padding='max_length',
    )
    logger.info(
        f"TiktokenPreprocessor: vocab_size={preprocessor.vocab_size}, "
        f"encoding={config.encoding_name}"
    )
    return preprocessor


def _decode_text(text) -> str:
    """Decode a TF text tensor to a Python string."""
    if isinstance(text, bytes):
        return text.decode('utf-8')
    if hasattr(text, 'numpy'):
        text_np = text.numpy()
        return text_np.decode('utf-8') if isinstance(text_np, bytes) else str(text_np)
    return str(text)


def load_dataset(config: TrainingConfig, split: str = "train") -> tf.data.Dataset:
    """Load text dataset from tensorflow-datasets."""
    logger.info(f"Loading {config.dataset_name} ({split})...")
    dataset, _ = tfds.load(
        config.dataset_name, split=split,
        as_supervised=False, shuffle_files=True, with_info=True,
    )
    dataset = dataset.map(lambda x: x["text"], num_parallel_calls=tf.data.AUTOTUNE)

    if config.max_samples is not None and split == "train":
        dataset = dataset.take(config.max_samples)
        logger.info(f"Limited training data to {config.max_samples} samples")
    elif split != "train":
        limit = config.max_samples // 5 if config.max_samples else 2000
        dataset = dataset.take(limit)
    return dataset


def preprocess_dataset(
    dataset: tf.data.Dataset,
    preprocessor: TiktokenPreprocessor,
    config: TrainingConfig,
) -> tf.data.Dataset:
    """Tokenize and batch text dataset for MLM training."""
    def tokenize_fn(text):
        encoded = preprocessor(_decode_text(text), return_tensors='np')
        return encoded['input_ids'][0], encoded['attention_mask'][0], encoded['token_type_ids'][0]

    dataset = dataset.map(
        lambda x: tf.py_function(tokenize_fn, [x], [tf.int32, tf.int32, tf.int32]),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    seq_len = config.max_seq_length
    dataset = dataset.map(
        lambda ids, mask, types: {
            'input_ids': tf.ensure_shape(ids, [seq_len]),
            'attention_mask': tf.ensure_shape(mask, [seq_len]),
            'token_type_ids': tf.ensure_shape(types, [seq_len]),
        },
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = (
        dataset.cache()
        .shuffle(buffer_size=1000)
        .batch(config.batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
    logger.info(f"Dataset preprocessed: batch_size={config.batch_size}")
    return dataset


# ---------------------------------------------------------------------
# Model Creation
# ---------------------------------------------------------------------


def create_bert_mlm_model(config: TrainingConfig) -> MaskedLanguageModel:
    """Create BERT encoder wrapped in MaskedLanguageModel."""
    logger.info(f"Creating BERT-{config.bert_variant.upper()} encoder...")
    encoder = BERT.from_variant(
        variant=config.bert_variant,
        vocab_size=config.vocab_size,
        max_position_embeddings=config.max_seq_length,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )

    special_token_ids = [
        config.cls_token_id, config.sep_token_id,
        config.pad_token_id, config.mask_token_id,
    ]
    mlm_model = MaskedLanguageModel(
        encoder=encoder,
        vocab_size=config.vocab_size,
        mask_ratio=config.mask_ratio,
        mask_token_id=config.mask_token_id,
        random_token_ratio=config.random_token_ratio,
        unchanged_ratio=config.unchanged_ratio,
        special_token_ids=special_token_ids,
        mlm_head_activation="gelu",
        initializer_range=0.02,
        mlm_head_dropout=0.1,
        layer_norm_eps=1e-12,
    )

    # Build to count parameters
    dummy = {k: tf.ones((1, config.max_seq_length), dtype=tf.int32)
             for k in ('input_ids', 'attention_mask')}
    dummy['token_type_ids'] = tf.zeros((1, config.max_seq_length), dtype=tf.int32)
    _ = mlm_model(dummy, training=False)

    enc_p, total_p = encoder.count_params(), mlm_model.count_params()
    logger.info(f"MLM model: {total_p:,} params ({enc_p:,} encoder + {total_p - enc_p:,} head)")
    return mlm_model


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------


def create_learning_rate_schedule(config: TrainingConfig, steps_per_epoch: int):
    """Warmup + cosine decay schedule."""
    total_steps = config.num_epochs * steps_per_epoch
    warmup_steps = int(config.warmup_ratio * total_steps)
    primary = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=config.learning_rate,
        decay_steps=total_steps - warmup_steps, alpha=0.0,
    )
    return WarmupSchedule(
        warmup_steps=warmup_steps, primary_schedule=primary, warmup_start_lr=1e-7,
    )


def compile_model(mlm_model: MaskedLanguageModel, config: TrainingConfig, steps_per_epoch: int):
    """Compile MLM model with AdamW and warmup schedule."""
    lr_schedule = create_learning_rate_schedule(config, steps_per_epoch)
    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule, weight_decay=config.weight_decay, clipnorm=1.0,
    )
    mlm_model.compile(optimizer=optimizer)
    logger.info(f"Compiled: AdamW, peak_lr={config.learning_rate}, wd={config.weight_decay}")


def create_callbacks(config: TrainingConfig) -> List[keras.callbacks.Callback]:
    """Create training callbacks."""
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    best_path = os.path.join(config.checkpoint_dir, "best_model.keras")

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=best_path, monitor="val_loss", mode="min",
            save_best_only=True, verbose=1,
        ),
        keras.callbacks.TensorBoard(
            log_dir=config.log_dir, histogram_freq=1, write_graph=True, update_freq='epoch',
        ),
        keras.callbacks.CSVLogger(os.path.join(config.save_dir, "training_log.csv"), append=True),
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: logger.info(
                f"Epoch {epoch + 1}: loss={logs['loss']:.4f}, acc={logs['accuracy']:.4f}, "
                f"val_loss={logs.get('val_loss', 'N/A'):.4f}, val_acc={logs.get('val_accuracy', 'N/A'):.4f}"
            )
        ),
    ]
    if config.run_epoch_analysis:
        callbacks.append(EpochAnalyzerCallback(
            output_dir=config.analysis_dir,
            start_epoch=config.analysis_start_epoch,
            epoch_frequency=config.analysis_epoch_frequency,
            model_name=f"BERT-{config.bert_variant}",
        ))
    logger.info(f"Created {len(callbacks)} callbacks")
    return callbacks


def train_bert_mlm(config: TrainingConfig) -> Tuple[MaskedLanguageModel, keras.callbacks.History]:
    """Run BERT MLM pre-training."""
    logger.info("=" * 60)
    logger.info("BERT MLM Pre-training with Tiktoken")
    logger.info("=" * 60)

    tf.random.set_seed(42)
    keras.utils.set_random_seed(42)
    os.makedirs(config.save_dir, exist_ok=True)

    preprocessor = create_tokenizer(config)
    train_dataset = preprocess_dataset(load_dataset(config, "train"), preprocessor, config)
    val_dataset = preprocess_dataset(load_dataset(config, "test"), preprocessor, config)

    steps_per_epoch = config.max_samples // config.batch_size if config.max_samples else 1000
    mlm_model = create_bert_mlm_model(config)
    compile_model(mlm_model, config, steps_per_epoch)
    callbacks = create_callbacks(config)

    logger.info("Starting training...")
    history = mlm_model.fit(
        train_dataset, epochs=config.num_epochs,
        callbacks=callbacks, validation_data=val_dataset, verbose=1,
    )
    logger.info("Training completed!")

    # Save full MLM model and encoder separately
    final_path = os.path.join(config.save_dir, "bert_mlm_final_best.keras")
    mlm_model.save(final_path)
    encoder_path = os.path.join(config.save_dir, "pretrained_bert_encoder_best.keras")
    mlm_model.encoder.save(encoder_path)

    # Summary
    best_epoch = tf.argmin(history.history['val_loss']).numpy()
    logger.info(
        f"Best epoch: {best_epoch + 1} "
        f"(val_loss: {history.history['val_loss'][best_epoch]:.4f}, "
        f"val_acc: {history.history['val_accuracy'][best_epoch]:.4f})"
    )
    logger.info(f"Models saved to: {config.save_dir}")
    return mlm_model, history


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------


def evaluate_model(
    mlm_model: MaskedLanguageModel,
    preprocessor: TiktokenPreprocessor,
    config: TrainingConfig,
) -> None:
    """Evaluate the trained model with MLM prediction visualization."""
    test_texts = [
        "The movie was really good and entertaining.",
        "I loved the acting and the storyline was amazing.",
        "This film was terrible and boring.",
        "The plot was confusing but the effects were great.",
    ]
    test_inputs = preprocessor.batch_encode(test_texts, return_tensors='np')
    test_batch = {k: tf.constant(v, dtype=tf.int32) for k, v in test_inputs.items()}
    visualize_mlm_predictions(mlm_model=mlm_model, inputs=test_batch, tokenizer=preprocessor, num_samples=4)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    """Main entry point for BERT MLM pre-training."""
    parser = argparse.ArgumentParser(description="BERT MLM Pre-training")
    parser.add_argument('--gpu', type=int, default=None, help='GPU device index')
    parser.add_argument('--variant', type=str, default='tiny', help='BERT variant')
    parser.add_argument('--epochs', type=int, default=3, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--max-samples', type=int, default=10000, help='Max training samples')
    args = parser.parse_args()

    setup_gpu(gpu_id=args.gpu)

    config = TrainingConfig()
    config.bert_variant = args.variant
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.max_samples = args.max_samples

    logger.info(f"Config: variant={config.bert_variant}, epochs={config.num_epochs}, "
                f"batch_size={config.batch_size}, lr={config.learning_rate}, "
                f"max_samples={config.max_samples}")

    mlm_model, history = train_bert_mlm(config)

    preprocessor = create_tokenizer(config)
    evaluate_model(mlm_model, preprocessor, config)

    logger.info(f"Pre-training complete! Encoder: {config.save_dir}/pretrained_bert_encoder_best.keras")


if __name__ == "__main__":
    main()
