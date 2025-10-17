"""BERT Pre-training Script with Masked Language Modeling.

A complete training script for pre-training BERT foundation models using
the Masked Language Modeling (MLM) objective on a small text dataset.

This script demonstrates the full workflow:
1. Loading and preprocessing text data from tensorflow-datasets
2. Creating tokenization pipeline with TiktokenPreprocessor
3. Building BERT encoder and MLM wrapper
4. Configuring training with learning rate schedules and callbacks
5. Pre-training the model
6. Saving the pretrained encoder for downstream tasks

The script uses the IMDB reviews dataset as a small, readily available
corpus for demonstration. For production, replace with larger datasets
like Wikipedia or BookCorpus.
"""

import os
import keras
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.models.bert import BERT
from dl_techniques.models.masked_language_model import (
    MaskedLanguageModel,
    visualize_mlm_predictions
)
from dl_techniques.utils.logger import logger
from dl_techniques.utils.tokenizer import TiktokenPreprocessor
from dl_techniques.optimization.warmup_schedule import WarmupSchedule
from dl_techniques.callbacks.analyzer_callback import EpochAnalyzerCallback

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


class TrainingConfig:
    """Configuration for BERT MLM pre-training.

    :param bert_variant: BERT model variant ('tiny', 'small', 'base', 'large').
    :type bert_variant: str
    :param vocab_size: Vocabulary size (for Tiktoken cl100k_base: 100277).
    :type vocab_size: int
    :param max_seq_length: Maximum sequence length for training.
    :type max_seq_length: int
    :param batch_size: Training batch size.
    :type batch_size: int
    :param num_epochs: Number of training epochs.
    :type num_epochs: int
    :param learning_rate: Peak learning rate.
    :type learning_rate: float
    :param warmup_ratio: Fraction of steps for learning rate warmup.
    :type warmup_ratio: float
    :param weight_decay: Weight decay for AdamW optimizer.
    :type weight_decay: float
    :param mask_ratio: Fraction of tokens to mask for MLM.
    :type mask_ratio: float
    :param save_dir: Directory to save checkpoints and final model.
    :type save_dir: str
    :param log_dir: Directory for TensorBoard logs.
    :type log_dir: str
    :param max_samples: Maximum number of training samples (None for all).
    :type max_samples: Optional[int]
    """

    # Model configuration
    bert_variant: str = "tiny"  # Use 'tiny' for quick testing
    vocab_size: int = 100277  # Tiktoken cl100k_base vocab size
    max_seq_length: int = 128

    # Tokenizer configuration
    encoding_name: str = "cl100k_base"  # Tiktoken encoding
    cls_token_id: int = 100264  # [CLS] token ID for Tiktoken
    sep_token_id: int = 100265  # [SEP] token ID for Tiktoken
    pad_token_id: int = 100266  # [PAD] token ID for Tiktoken
    mask_token_id: int = 100267  # [MASK] token ID for Tiktoken

    # Training configuration
    batch_size: int = 32
    num_epochs: int = 3
    learning_rate: float = 5e-4  # Higher LR for tiny model
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

    # MLM configuration
    mask_ratio: float = 0.15
    random_token_ratio: float = 0.1
    unchanged_ratio: float = 0.1

    # Paths
    save_dir: str = "results/bert_pretrain"
    log_dir: str = "results/bert_pretrain/logs"
    checkpoint_dir: str = "results/bert_pretrain/checkpoints"
    analysis_dir: str = "results/bert_pretrain/epoch_analysis"

    # Data configuration
    dataset_name: str = "imdb_reviews"
    max_samples: Optional[int] = 10000  # Limit for quick testing

    # In-Training Analysis Configuration
    run_epoch_analysis: bool = True  # Master switch
    analysis_start_epoch: int = 1
    analysis_epoch_frequency: int = 5


# ---------------------------------------------------------------------
# Data Pipeline
# ---------------------------------------------------------------------

def create_tokenizer(config: TrainingConfig) -> TiktokenPreprocessor:
    """Create and configure Tiktoken preprocessor.

    :param config: Training configuration.
    :type config: TrainingConfig
    :return: Configured TiktokenPreprocessor instance.
    :rtype: TiktokenPreprocessor
    """
    logger.info("Initializing TiktokenPreprocessor...")
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
        f"TiktokenPreprocessor initialized: "
        f"vocab_size={preprocessor.vocab_size}, "
        f"encoding={config.encoding_name}"
    )
    logger.info(
        f"Special tokens: [CLS]={config.cls_token_id}, "
        f"[SEP]={config.sep_token_id}, "
        f"[PAD]={config.pad_token_id}, "
        f"[MASK]={config.mask_token_id}"
    )
    return preprocessor


def load_dataset(
    config: TrainingConfig,
    split: str = "train"
) -> tf.data.Dataset:
    """Load text dataset from tensorflow-datasets.

    :param config: Training configuration.
    :type config: TrainingConfig
    :param split: Dataset split to load ('train', 'test', etc.).
    :type split: str
    :return: TensorFlow dataset.
    :rtype: tf.data.Dataset
    """
    logger.info(
        f"Loading {config.dataset_name} dataset ({split} split)..."
    )

    # Load dataset
    dataset, info = tfds.load(
        config.dataset_name,
        split=split,
        as_supervised=False,
        shuffle_files=True,
        with_info=True,
    )

    # Extract text field (IMDB has 'text' field)
    dataset = dataset.map(
        lambda x: x["text"],
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Limit samples if specified
    if config.max_samples is not None and split == "train":
        dataset = dataset.take(config.max_samples)
        logger.info(
            f"Limited training data to {config.max_samples} samples"
        )
    elif split != "train":
        # Also limit validation data for faster evaluation cycles
        limit = (
            config.max_samples // 5 if config.max_samples else 2000
        )
        dataset = dataset.take(limit)
        logger.info("Limited validation data for faster evaluation")

    return dataset


def preprocess_dataset(
    dataset: tf.data.Dataset,
    preprocessor: TiktokenPreprocessor,
    config: TrainingConfig
) -> tf.data.Dataset:
    """Preprocess text dataset for MLM training.

    :param dataset: Raw text dataset.
    :type dataset: tf.data.Dataset
    :param preprocessor: Tiktoken preprocessor.
    :type preprocessor: TiktokenPreprocessor
    :param config: Training configuration.
    :type config: TrainingConfig
    :return: Preprocessed dataset ready for training.
    :rtype: tf.data.Dataset
    """
    logger.info("Preprocessing dataset with TiktokenPreprocessor...")

    def tokenize_function(
        text: tf.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Tokenize a single text sample using TiktokenPreprocessor.

        :param text: Input text string.
        :type text: tf.Tensor
        :return: Tuple of (input_ids, attention_mask, token_type_ids).
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        # Decode bytes to string if necessary
        if isinstance(text, bytes):
            text_str = text.decode('utf-8')
        elif hasattr(text, 'numpy'):
            text_np = text.numpy()
            if isinstance(text_np, bytes):
                text_str = text_np.decode('utf-8')
            else:
                text_str = str(text_np)
        else:
            text_str = str(text)

        # Tokenize using TiktokenPreprocessor
        # The preprocessor returns a dict with batch dimension (1, seq_len)
        encoded = preprocessor(text_str, return_tensors='np')

        # Extract and squeeze to remove batch dimension
        return (
            encoded['input_ids'][0],  # Shape: (seq_len,)
            encoded['attention_mask'][0],  # Shape: (seq_len,)
            encoded['token_type_ids'][0],  # Shape: (seq_len,)
        )

    # Apply tokenization - py_function returns a tuple
    dataset = dataset.map(
        lambda x: tf.py_function(
            tokenize_function,
            [x],
            [tf.int32, tf.int32, tf.int32]  # List of output types
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Convert tuple to dictionary and set shapes
    def tuple_to_dict(
        input_ids: tf.Tensor,
        attention_mask: tf.Tensor,
        token_type_ids: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        """Convert tuple output to dictionary format.

        :param input_ids: Token IDs.
        :type input_ids: tf.Tensor
        :param attention_mask: Attention mask.
        :type attention_mask: tf.Tensor
        :param token_type_ids: Token type IDs.
        :type token_type_ids: tf.Tensor
        :return: Dictionary with proper shapes.
        :rtype: Dict[str, tf.Tensor]
        """
        return {
            'input_ids': tf.ensure_shape(
                input_ids,
                [config.max_seq_length]
            ),
            'attention_mask': tf.ensure_shape(
                attention_mask,
                [config.max_seq_length]
            ),
            'token_type_ids': tf.ensure_shape(
                token_type_ids,
                [config.max_seq_length]
            ),
        }

    dataset = dataset.map(
        tuple_to_dict,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Cache the tokenized and processed dataset in memory
    dataset = dataset.cache()

    # Shuffle, batch, and prefetch
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(config.batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    logger.info(
        f"Dataset preprocessed and cached: "
        f"batch_size={config.batch_size}"
    )

    return dataset


# ---------------------------------------------------------------------
# Model Creation
# ---------------------------------------------------------------------


def create_bert_mlm_model(
    config: TrainingConfig,
) -> MaskedLanguageModel:
    """Create BERT encoder and MLM wrapper.

    :param config: Training configuration.
    :type config: TrainingConfig
    :return: MaskedLanguageModel ready for training.
    :rtype: MaskedLanguageModel
    """
    logger.info(
        f"Creating BERT-{config.bert_variant.upper()} encoder..."
    )

    # Create BERT encoder
    encoder = BERT.from_variant(
        variant=config.bert_variant,
        vocab_size=config.vocab_size,
        max_position_embeddings=config.max_seq_length,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )

    logger.info(
        f"BERT encoder created with hidden_size={encoder.hidden_size}"
    )

    # Create list of special token IDs for Tiktoken
    special_token_ids = [
        config.cls_token_id,
        config.sep_token_id,
        config.pad_token_id,
        config.mask_token_id,
    ]

    # Create MLM wrapper
    logger.info("Creating MaskedLanguageModel wrapper...")
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

    # Build the model to count parameters
    logger.info("Building model...")
    dummy_input = {
        'input_ids': tf.ones(
            (1, config.max_seq_length),
            dtype=tf.int32
        ),
        'attention_mask': tf.ones(
            (1, config.max_seq_length),
            dtype=tf.int32
        ),
        'token_type_ids': tf.zeros(
            (1, config.max_seq_length),
            dtype=tf.int32
        ),
    }
    _ = mlm_model(dummy_input, training=False)

    encoder_params = encoder.count_params()
    total_params = mlm_model.count_params()
    head_params = total_params - encoder_params

    logger.info(
        f"MLM model built: {total_params:,} total parameters "
        f"({encoder_params:,} encoder + {head_params:,} MLM head)"
    )

    return mlm_model


# ---------------------------------------------------------------------
# Training Configuration
# ---------------------------------------------------------------------


def create_learning_rate_schedule(
    config: TrainingConfig,
    steps_per_epoch: int
) -> keras.optimizers.schedules.LearningRateSchedule:
    """Create learning rate schedule with warmup and cosine decay.

    :param config: Training configuration.
    :type config: TrainingConfig
    :param steps_per_epoch: Number of training steps per epoch.
    :type steps_per_epoch: int
    :return: Learning rate schedule.
    :rtype: keras.optimizers.schedules.LearningRateSchedule
    """
    total_steps = config.num_epochs * steps_per_epoch
    warmup_steps = int(config.warmup_ratio * total_steps)

    logger.info(
        f"Learning rate schedule: total_steps={total_steps}, "
        f"warmup_steps={warmup_steps}"
    )

    # 1. Create the primary (post-warmup) schedule
    primary_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=config.learning_rate,
        decay_steps=total_steps - warmup_steps,
        alpha=0.0
    )

    # 2. Wrap it with the WarmupSchedule
    warmup_schedule = WarmupSchedule(
        warmup_steps=warmup_steps,
        primary_schedule=primary_schedule,
        warmup_start_lr=1e-7  # A small starting learning rate
    )

    return warmup_schedule


def compile_model(
    mlm_model: MaskedLanguageModel,
    config: TrainingConfig,
    steps_per_epoch: int
) -> None:
    """Compile MLM model with optimizer and metrics.

    :param mlm_model: MaskedLanguageModel instance.
    :type mlm_model: MaskedLanguageModel
    :param config: Training configuration.
    :type config: TrainingConfig
    :param steps_per_epoch: Number of training steps per epoch.
    :type steps_per_epoch: int
    """
    logger.info("Compiling model...")

    # Create learning rate schedule
    lr_schedule = create_learning_rate_schedule(config, steps_per_epoch)

    # Create optimizer
    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=config.weight_decay,
        clipnorm=1.0,  # Gradient clipping
    )

    # Compile model
    mlm_model.compile(optimizer=optimizer)

    logger.info(
        f"Model compiled: optimizer=AdamW, "
        f"peak_lr={config.learning_rate}, "
        f"weight_decay={config.weight_decay}"
    )


def create_callbacks(config: TrainingConfig) -> List[keras.callbacks.Callback]:
    """Create training callbacks.

    :param config: Training configuration.
    :type config: TrainingConfig
    :return: List of Keras callbacks.
    :rtype: List[keras.callbacks.Callback]
    """
    logger.info("Creating training callbacks...")

    # Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    # Define file path for the best model checkpoint
    best_model_filepath = os.path.join(
        config.checkpoint_dir,
        "best_model.keras"
    )

    callbacks = [
        # Model checkpointing
        keras.callbacks.ModelCheckpoint(
            filepath=best_model_filepath,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),

        # TensorBoard logging
        keras.callbacks.TensorBoard(
            log_dir=config.log_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch',
        ),

        # CSV logging
        keras.callbacks.CSVLogger(
            filename=os.path.join(config.save_dir, "training_log.csv"),
            append=True,
        ),

        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),

        # Learning rate logging
        keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: logger.info(
                f"Epoch {epoch + 1}: "
                f"loss={logs['loss']:.4f}, "
                f"accuracy={logs['accuracy']:.4f}, "
                f"val_loss={logs.get('val_loss', 'N/A'):.4f}, "
                f"val_accuracy={logs.get('val_accuracy', 'N/A'):.4f}"
            )
        ),
    ]

    # Conditionally add the EpochAnalyzerCallback
    if config.run_epoch_analysis:
        analysis_callback = EpochAnalyzerCallback(
            output_dir=config.analysis_dir,
            start_epoch=config.analysis_start_epoch,
            epoch_frequency=config.analysis_epoch_frequency,
            model_name=f"BERT-{config.bert_variant}"
        )
        callbacks.append(analysis_callback)

    logger.info(
        f"Created {len(callbacks)} callbacks. "
        f"Best model will be saved to {best_model_filepath}"
    )

    return callbacks


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------


def train_bert_mlm(
    config: TrainingConfig
) -> Tuple[MaskedLanguageModel, keras.callbacks.History]:
    """Main training function for BERT MLM pre-training.

    :param config: Training configuration.
    :type config: TrainingConfig
    :return: Tuple of (trained model, training history).
    :rtype: Tuple[MaskedLanguageModel, keras.callbacks.History]
    """
    logger.info("=" * 80)
    logger.info("BERT Masked Language Model Pre-training with Tiktoken")
    logger.info("=" * 80)

    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    keras.utils.set_random_seed(42)

    # Create output directory
    os.makedirs(config.save_dir, exist_ok=True)

    # Step 1: Create tokenizer
    preprocessor = create_tokenizer(config)

    # Step 2: Load and preprocess datasets
    raw_train_dataset = load_dataset(config, split="train")
    train_dataset = preprocess_dataset(
        raw_train_dataset,
        preprocessor,
        config
    )

    raw_val_dataset = load_dataset(config, split="test")
    val_dataset = preprocess_dataset(
        raw_val_dataset,
        preprocessor,
        config
    )

    # Calculate steps per epoch
    steps_per_epoch = (
        config.max_samples // config.batch_size
        if config.max_samples else 1000
    )
    logger.info(f"Estimated steps per epoch: {steps_per_epoch}")

    # Step 3: Create model
    mlm_model = create_bert_mlm_model(config)

    # Step 4: Compile model
    compile_model(mlm_model, config, steps_per_epoch)

    # Step 5: Create callbacks
    callbacks = create_callbacks(config)

    # Step 6: Train model
    logger.info("=" * 80)
    logger.info("Starting training...")
    logger.info("=" * 80)

    history = mlm_model.fit(
        train_dataset,
        epochs=config.num_epochs,
        callbacks=callbacks,
        validation_data=val_dataset,
        verbose=1,
    )

    logger.info("=" * 80)
    logger.info("Training completed!")
    logger.info("=" * 80)

    # Step 7: Save final model (which holds the best weights)
    final_model_path = os.path.join(
        config.save_dir,
        "bert_mlm_final_best.keras"
    )
    logger.info(
        f"Saving final model with best weights to {final_model_path}"
    )
    mlm_model.save(final_model_path)

    # Step 8: Save pretrained encoder separately
    encoder_path = os.path.join(
        config.save_dir,
        "pretrained_bert_encoder_best.keras"
    )
    logger.info(f"Saving best pretrained encoder to {encoder_path}")
    mlm_model.encoder.save(encoder_path)

    # Print summary statistics
    logger.info("=" * 80)
    logger.info("Training Summary")
    logger.info("=" * 80)
    logger.info(f"Total epochs run: {len(history.history['loss'])}")

    # Find the best epoch results
    best_epoch = tf.argmin(history.history['val_loss']).numpy()
    best_val_loss = history.history['val_loss'][best_epoch]
    best_val_acc = history.history['val_accuracy'][best_epoch]
    logger.info(
        f"Best epoch: {best_epoch + 1} "
        f"(val_loss: {best_val_loss:.4f}, "
        f"val_accuracy: {best_val_acc:.4f})"
    )
    logger.info(f"Model with best weights saved to: {config.save_dir}")
    logger.info("=" * 80)

    return mlm_model, history


# ---------------------------------------------------------------------
# Evaluation and Visualization
# ---------------------------------------------------------------------


def evaluate_model(
    mlm_model: MaskedLanguageModel,
    preprocessor: TiktokenPreprocessor,
    config: TrainingConfig
) -> None:
    """Evaluate the trained model and visualize predictions.

    :param mlm_model: Trained MaskedLanguageModel.
    :type mlm_model: MaskedLanguageModel
    :param preprocessor: Tiktoken preprocessor.
    :type preprocessor: TiktokenPreprocessor
    :param config: Training configuration.
    :type config: TrainingConfig
    """
    logger.info("=" * 80)
    logger.info("Model Evaluation")
    logger.info("=" * 80)

    # Create a small test dataset
    test_texts = [
        "The movie was really good and entertaining.",
        "I loved the acting and the storyline was amazing.",
        "This film was terrible and boring.",
        "The plot was confusing but the effects were great.",
    ]

    # Tokenize test samples using TiktokenPreprocessor
    test_inputs = preprocessor.batch_encode(
        test_texts,
        return_tensors='np'
    )

    # Convert to dict of TensorFlow tensors
    test_batch = {
        'input_ids': tf.constant(
            test_inputs['input_ids'],
            dtype=tf.int32
        ),
        'attention_mask': tf.constant(
            test_inputs['attention_mask'],
            dtype=tf.int32
        ),
        'token_type_ids': tf.constant(
            test_inputs['token_type_ids'],
            dtype=tf.int32
        ),
    }

    # Visualize predictions
    # Note: visualize_mlm_predictions expects a tokenizer with decode method
    # TiktokenPreprocessor has a decode method, so it should work
    visualize_mlm_predictions(
        mlm_model=mlm_model,
        inputs=test_batch,
        tokenizer=preprocessor,
        num_samples=4
    )


# ---------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------


def main() -> None:
    """Main entry point for BERT MLM pre-training script."""
    # Create configuration
    config = TrainingConfig()

    # Log configuration
    logger.info("Training Configuration:")
    logger.info(f"  - BERT variant: {config.bert_variant}")
    logger.info(f"  - Tokenizer: TiktokenPreprocessor")
    logger.info(f"  - Encoding: {config.encoding_name}")
    logger.info(f"  - Vocabulary size: {config.vocab_size}")
    logger.info(f"  - Max sequence length: {config.max_seq_length}")
    logger.info(f"  - Batch size: {config.batch_size}")
    logger.info(f"  - Number of epochs: {config.num_epochs}")
    logger.info(f"  - Learning rate: {config.learning_rate}")
    logger.info(f"  - Mask ratio: {config.mask_ratio}")
    logger.info(f"  - Dataset: {config.dataset_name}")
    logger.info(f"  - Max samples: {config.max_samples}")
    logger.info(f"  - Save directory: {config.save_dir}")

    # Train model
    mlm_model, history = train_bert_mlm(config)

    # Evaluate model
    preprocessor = create_tokenizer(config)
    evaluate_model(mlm_model, preprocessor, config)

    logger.info("=" * 80)
    logger.info(
        "Pre-training complete! Encoder ready for fine-tuning."
    )
    logger.info(
        f"Load encoder: keras.models.load_model("
        f"'{config.save_dir}/pretrained_bert_encoder_best.keras')"
    )
    logger.info("=" * 80)


if __name__ == "__main__":
    main()