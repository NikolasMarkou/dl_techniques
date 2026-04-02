"""GPT-2 Pre-training Script with Causal Language Modeling.

Pre-trains a GPT-2 decoder on a text dataset (IMDB reviews by default)
using next-token prediction (causal LM). Saves the trained model for
downstream fine-tuning or text generation.
"""

import argparse
import os

import keras
import tensorflow as tf
from typing import Optional, Tuple

from train.common import setup_gpu
from train.common.nlp import (
    create_tokenizer,
    load_text_dataset,
    preprocess_clm_dataset,
    create_warmup_lr_schedule,
    create_nlp_callbacks,
)

from dl_techniques.models.gpt2 import GPT2
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


class TrainingConfig:
    """Configuration for GPT-2 CLM pre-training."""

    # Model
    gpt2_variant: str = "tiny"
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

    # Paths
    save_dir: str = "results/gpt2_pretrain"

    # Data
    dataset_name: str = "imdb_reviews"
    max_samples: Optional[int] = 10000

    # Analysis
    run_epoch_analysis: bool = True
    analysis_start_epoch: int = 1
    analysis_epoch_frequency: int = 5


# ---------------------------------------------------------------------
# Model Creation
# ---------------------------------------------------------------------


def create_gpt2_model(config: TrainingConfig) -> GPT2:
    """Create GPT-2 model from variant configuration."""
    logger.info(f"Creating GPT-2-{config.gpt2_variant.upper()}...")
    model = GPT2.from_variant(
        variant=config.gpt2_variant,
        vocab_size=config.vocab_size,
        max_seq_len=config.max_seq_length,
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
    )

    # Build to count parameters
    dummy = keras.random.uniform(
        (1, config.max_seq_length - 1), 0, config.vocab_size, dtype="int32"
    )
    _ = model(dummy, training=False)

    total_p = model.count_params()
    logger.info(f"GPT-2 model: {total_p:,} parameters")
    return model


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------


def compile_model(
    model: GPT2,
    config: TrainingConfig,
    steps_per_epoch: int,
) -> None:
    """Compile GPT-2 with AdamW and warmup schedule."""
    lr_schedule = create_warmup_lr_schedule(
        config.learning_rate,
        config.num_epochs,
        steps_per_epoch,
        config.warmup_ratio,
    )
    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=config.weight_decay,
        clipnorm=1.0,
    )
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    logger.info(
        f"Compiled: AdamW, peak_lr={config.learning_rate}, "
        f"wd={config.weight_decay}"
    )


def train_gpt2(
    config: TrainingConfig,
) -> Tuple[GPT2, keras.callbacks.History]:
    """Run GPT-2 CLM pre-training."""
    logger.info("=" * 60)
    logger.info("GPT-2 Causal LM Pre-training")
    logger.info("=" * 60)

    tf.random.set_seed(42)
    keras.utils.set_random_seed(42)
    os.makedirs(config.save_dir, exist_ok=True)

    # Tokenizer
    preprocessor = create_tokenizer(
        config.encoding_name,
        config.max_seq_length,
        config.cls_token_id,
        config.sep_token_id,
        config.pad_token_id,
        config.mask_token_id,
    )

    # Data
    train_dataset = preprocess_clm_dataset(
        load_text_dataset(config.dataset_name, "train", config.max_samples),
        preprocessor,
        config.max_seq_length,
        config.batch_size,
    )
    val_dataset = preprocess_clm_dataset(
        load_text_dataset(config.dataset_name, "test", config.max_samples),
        preprocessor,
        config.max_seq_length,
        config.batch_size,
    )

    # Model
    steps_per_epoch = (
        config.max_samples // config.batch_size
        if config.max_samples
        else 1000
    )
    model = create_gpt2_model(config)
    compile_model(model, config, steps_per_epoch)

    # Callbacks
    callbacks, results_dir = create_nlp_callbacks(
        model_name=f"GPT2-{config.gpt2_variant}",
        results_dir_prefix="gpt2_pretrain",
        include_analyzer=config.run_epoch_analysis,
        analyzer_epoch_frequency=config.analysis_epoch_frequency,
        analyzer_start_epoch=config.analysis_start_epoch,
    )

    # Train
    logger.info("Starting training...")
    history = model.fit(
        train_dataset,
        epochs=config.num_epochs,
        callbacks=callbacks,
        validation_data=val_dataset,
        verbose=1,
    )
    logger.info("Training completed!")

    # Save
    final_path = os.path.join(config.save_dir, "gpt2_final_best.keras")
    model.save(final_path)
    logger.info(f"Model saved to: {final_path}")

    # Summary
    best_epoch = tf.argmin(history.history["val_loss"]).numpy()
    logger.info(
        f"Best epoch: {best_epoch + 1} "
        f"(val_loss: {history.history['val_loss'][best_epoch]:.4f})"
    )
    return model, history


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    """Main entry point for GPT-2 CLM pre-training."""
    parser = argparse.ArgumentParser(description="GPT-2 CLM Pre-training")
    parser.add_argument(
        "--gpu", type=int, default=None, help="GPU device index"
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="tiny",
        choices=list(GPT2.MODEL_VARIANTS.keys()),
        help="GPT-2 variant",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10000,
        help="Max training samples",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=128,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-4,
        help="Peak learning rate",
    )
    args = parser.parse_args()

    setup_gpu(gpu_id=args.gpu)

    config = TrainingConfig()
    config.gpt2_variant = args.variant
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.max_samples = args.max_samples
    config.max_seq_length = args.max_seq_length
    config.learning_rate = args.learning_rate

    logger.info(
        f"Config: variant={config.gpt2_variant}, epochs={config.num_epochs}, "
        f"batch_size={config.batch_size}, lr={config.learning_rate}, "
        f"max_samples={config.max_samples}, seq_len={config.max_seq_length}"
    )

    model, history = train_gpt2(config)
    logger.info(
        f"Pre-training complete! Model: {config.save_dir}/gpt2_final_best.keras"
    )


if __name__ == "__main__":
    main()
