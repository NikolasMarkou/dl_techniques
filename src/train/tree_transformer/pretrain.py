"""Tree Transformer Pre-training Script with Masked Language Modeling.

Pre-trains a TreeTransformer encoder using MLM on a text dataset (IMDB reviews
by default). Saves both the full MLM model and the encoder separately for
downstream fine-tuning. Mirrors src/train/bert/pretrain.py (Pattern 3 NLP).

CRITICAL: `pad_token_id` MUST match the tokenizer's pad id (100266 for
tiktoken cl100k_base). Leaving the TreeTransformer default of 0 will silently
mis-mask every input.
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
    preprocess_mlm_dataset,
    create_warmup_lr_schedule,
    create_nlp_callbacks,
)

from dl_techniques.models.tree_transformer import TreeTransformer
from dl_techniques.models.masked_language_model import MaskedLanguageModel
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


class TrainingConfig:
    """Configuration for TreeTransformer MLM pre-training."""

    # Model
    variant: str = "tiny"
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
    save_dir: str = "results/tree_transformer_pretrain"

    # Data
    dataset_name: str = "imdb_reviews"
    max_samples: Optional[int] = 10000

    # Analysis
    run_epoch_analysis: bool = False  # off by default for fast smoke
    analysis_start_epoch: int = 1
    analysis_epoch_frequency: int = 5


# ---------------------------------------------------------------------
# Model Creation
# ---------------------------------------------------------------------


def create_tree_transformer_mlm_model(config: TrainingConfig) -> MaskedLanguageModel:
    """Create TreeTransformer encoder wrapped in MaskedLanguageModel."""
    logger.info(f"Creating TreeTransformer-{config.variant.upper()} encoder...")
    # CRITICAL: pad_token_id must match the tokenizer's pad id.
    encoder = TreeTransformer.from_variant(
        config.variant,
        vocab_size=config.vocab_size,
        max_len=config.max_seq_length,
        pad_token_id=config.pad_token_id,
        hidden_dropout_rate=0.1,
        attention_dropout_rate=0.1,
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
    dummy = {
        k: tf.ones((1, config.max_seq_length), dtype=tf.int32)
        for k in ("input_ids", "attention_mask")
    }
    dummy["token_type_ids"] = tf.zeros((1, config.max_seq_length), dtype=tf.int32)
    _ = mlm_model(dummy, training=False)

    enc_p, total_p = encoder.count_params(), mlm_model.count_params()
    logger.info(
        f"MLM model: {total_p:,} params ({enc_p:,} encoder + "
        f"{total_p - enc_p:,} head)"
    )
    return mlm_model


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------


def compile_model(
    mlm_model: MaskedLanguageModel,
    config: TrainingConfig,
    steps_per_epoch: int,
) -> None:
    """Compile MLM model with AdamW and warmup schedule.

    `clipnorm=1.0` is recommended per README §13: the `log`/`exp` ops in
    GroupAttention are sensitive to large gradients.
    """
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
    mlm_model.compile(optimizer=optimizer)
    logger.info(
        f"Compiled: AdamW, peak_lr={config.learning_rate}, "
        f"wd={config.weight_decay}, clipnorm=1.0"
    )


def train_tree_transformer_mlm(
    config: TrainingConfig,
) -> Tuple[MaskedLanguageModel, keras.callbacks.History]:
    """Run TreeTransformer MLM pre-training."""
    logger.info("=" * 60)
    logger.info("Tree Transformer MLM Pre-training with Tiktoken")
    logger.info("=" * 60)

    tf.random.set_seed(42)
    keras.utils.set_random_seed(42)
    os.makedirs(config.save_dir, exist_ok=True)

    preprocessor = create_tokenizer(
        config.encoding_name, config.max_seq_length,
        config.cls_token_id, config.sep_token_id,
        config.pad_token_id, config.mask_token_id,
    )
    train_dataset = preprocess_mlm_dataset(
        load_text_dataset(config.dataset_name, "train", config.max_samples),
        preprocessor, config.max_seq_length, config.batch_size,
    )
    val_dataset = preprocess_mlm_dataset(
        load_text_dataset(config.dataset_name, "test", config.max_samples),
        preprocessor, config.max_seq_length, config.batch_size,
    )

    steps_per_epoch = max(
        1,
        (config.max_samples or 10000) // config.batch_size,
    )
    mlm_model = create_tree_transformer_mlm_model(config)
    compile_model(mlm_model, config, steps_per_epoch)
    callbacks, results_dir = create_nlp_callbacks(
        model_name=f"TreeTransformer-{config.variant}",
        results_dir_prefix="tree_transformer_pretrain",
        include_analyzer=config.run_epoch_analysis,
        analyzer_epoch_frequency=config.analysis_epoch_frequency,
        analyzer_start_epoch=config.analysis_start_epoch,
    )

    logger.info("Starting training...")
    history = mlm_model.fit(
        train_dataset, epochs=config.num_epochs,
        callbacks=callbacks, validation_data=val_dataset, verbose=1,
    )
    logger.info("Training completed!")

    # Save full MLM model and encoder separately
    final_path = os.path.join(results_dir, "tree_transformer_mlm_final_best.keras")
    mlm_model.save(final_path)
    encoder_path = os.path.join(
        results_dir, "pretrained_tree_transformer_encoder_best.keras"
    )
    mlm_model.encoder.save(encoder_path)

    logger.info(f"Models saved to: {results_dir}")
    return mlm_model, history


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    """Main entry point for TreeTransformer MLM pre-training."""
    parser = argparse.ArgumentParser(description="TreeTransformer MLM Pre-training")
    parser.add_argument("--gpu", type=int, default=None, help="GPU device index")
    parser.add_argument("--variant", type=str, default="tiny",
                        help="TreeTransformer variant")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--max-samples", type=int, default=10000,
                        help="Max training samples")
    parser.add_argument("--max-seq-length", type=int, default=128,
                        help="Max sequence length")
    parser.add_argument("--learning-rate", type=float, default=5e-4,
                        help="Peak learning rate")
    args = parser.parse_args()

    setup_gpu(gpu_id=args.gpu)

    config = TrainingConfig()
    config.variant = args.variant
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.max_samples = args.max_samples
    config.max_seq_length = args.max_seq_length
    config.learning_rate = args.learning_rate

    logger.info(
        f"Config: variant={config.variant}, epochs={config.num_epochs}, "
        f"batch_size={config.batch_size}, lr={config.learning_rate}, "
        f"max_samples={config.max_samples}, "
        f"max_seq_length={config.max_seq_length}"
    )

    mlm_model, history = train_tree_transformer_mlm(config)

    logger.info(
        f"Pre-training complete! Encoder saved alongside the full MLM model in "
        f"{config.save_dir}/."
    )


if __name__ == "__main__":
    main()
