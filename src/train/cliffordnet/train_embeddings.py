"""CliffordNet Embedding (Bidirectional U-Net) MLM Pre-training Script.

Pre-trains a :class:`CliffordNetEmbedding` encoder via Masked Language Modeling
on a text dataset (IMDB reviews by default). Mirrors `train/bert/pretrain.py`
~1-for-1 with surgical encoder-construction substitutions. Saves both the full
MLM model and the encoder separately for downstream fine-tuning.

Usage:
    MPLBACKEND=Agg python -m train.cliffordnet.train_embeddings --variant nano
    MPLBACKEND=Agg python -m train.cliffordnet.train_embeddings --smoke

Pattern-3 NLP pretrain script (see src/train/CLAUDE.md).
"""

import argparse
import os
from typing import Optional, Tuple

import keras
import tensorflow as tf

from train.common import setup_gpu, set_seeds
from train.common.nlp import (
    create_nlp_callbacks,
    create_tokenizer,
    create_warmup_lr_schedule,
    load_text_dataset,
    preprocess_mlm_dataset,
)

from dl_techniques.models.cliffordnet import CliffordNetEmbedding
from dl_techniques.models.masked_language_model import (
    MaskedLanguageModel,
    visualize_mlm_predictions,
)
from dl_techniques.utils.logger import logger
from dl_techniques.utils.tokenizer import TiktokenPreprocessor


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


class TrainingConfig:
    """Configuration for CliffordNetEmbedding MLM pre-training."""

    # Model
    variant: str = "nano"
    vocab_size: int = 100277  # tiktoken cl100k_base
    max_seq_length: int = 128
    pooling_strategy: str = "mean"

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
    save_dir: str = "results/cliffordnet_embedding_pretrain"

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


def create_cliffordnet_embedding_mlm_model(
    config: TrainingConfig,
) -> MaskedLanguageModel:
    """Create CliffordNetEmbedding encoder wrapped in MaskedLanguageModel."""
    logger.info(f"Creating CliffordNetEmbedding-{config.variant.upper()} encoder...")
    # NOTE: pad_token_id=config.pad_token_id is explicitly wired into the
    # encoder ctor (LESSONS hard rule for tokenizer-aware encoders).
    encoder = CliffordNetEmbedding.from_variant(
        variant=config.variant,
        vocab_size=config.vocab_size,
        max_seq_length=config.max_seq_length,
        pooling_strategy=config.pooling_strategy,
        pad_token_id=config.pad_token_id,
        dropout_rate=0.1,
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

    # Build to count parameters.
    dummy = {
        "input_ids": tf.ones((1, config.max_seq_length), dtype=tf.int32),
        "attention_mask": tf.ones((1, config.max_seq_length), dtype=tf.int32),
    }
    _ = mlm_model(dummy, training=False)

    enc_p = encoder.count_params()
    total_p = mlm_model.count_params()
    logger.info(
        f"MLM model: {total_p:,} params "
        f"({enc_p:,} encoder + {total_p - enc_p:,} head)"
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
    """Compile MLM model with AdamW + warmup schedule (no L2 — LESSONS L72)."""
    lr_schedule = create_warmup_lr_schedule(
        config.learning_rate, config.num_epochs, steps_per_epoch, config.warmup_ratio,
    )
    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule, weight_decay=config.weight_decay, clipnorm=1.0,
    )
    mlm_model.compile(optimizer=optimizer)
    logger.info(
        f"Compiled: AdamW, peak_lr={config.learning_rate}, "
        f"wd={config.weight_decay}"
    )


def train_cliffordnet_embedding_mlm(
    config: TrainingConfig,
) -> Tuple[MaskedLanguageModel, keras.callbacks.History]:
    """Run CliffordNetEmbedding MLM pre-training."""
    logger.info("=" * 60)
    logger.info("CliffordNetEmbedding MLM Pre-training with Tiktoken")
    logger.info("=" * 60)

    set_seeds(42)
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

    steps_per_epoch = (
        config.max_samples // config.batch_size if config.max_samples else 1000
    )
    mlm_model = create_cliffordnet_embedding_mlm_model(config)
    compile_model(mlm_model, config, steps_per_epoch)
    callbacks, results_dir = create_nlp_callbacks(
        model_name=f"CliffordNetEmbedding-{config.variant}",
        results_dir_prefix="cliffordnet_embedding_pretrain",
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

    # Save full MLM model + encoder separately. Mirrors BERT trainer
    # convention so follow-up fine-tune scripts can do
    # `keras.models.load_model(encoder_path,
    #     custom_objects={"CliffordNetEmbedding": CliffordNetEmbedding})`.
    final_path = os.path.join(results_dir, "cliffordnet_embedding_mlm_final_best.keras")
    mlm_model.save(final_path)
    encoder_path = os.path.join(
        results_dir, "pretrained_cliffordnet_embedding_encoder_best.keras",
    )
    mlm_model.encoder.save(encoder_path)

    # Summary.
    if "val_loss" in history.history and len(history.history["val_loss"]) > 0:
        best_epoch = int(tf.argmin(history.history["val_loss"]).numpy())
        val_acc = history.history.get("val_accuracy", [float("nan")])
        logger.info(
            f"Best epoch: {best_epoch + 1} "
            f"(val_loss: {history.history['val_loss'][best_epoch]:.4f}, "
            f"val_acc: {val_acc[best_epoch]:.4f})"
        )
    logger.info(f"Models saved to: {results_dir}")
    return mlm_model, history


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------


def evaluate_model(
    mlm_model: MaskedLanguageModel,
    preprocessor: TiktokenPreprocessor,
) -> None:
    """Evaluate the trained model with MLM prediction visualization."""
    test_texts = [
        "The movie was really good and entertaining.",
        "I loved the acting and the storyline was amazing.",
        "This film was terrible and boring.",
        "The plot was confusing but the effects were great.",
    ]
    test_inputs = preprocessor.batch_encode(test_texts, return_tensors="np")
    test_batch = {k: tf.constant(v, dtype=tf.int32) for k, v in test_inputs.items()}
    visualize_mlm_predictions(
        mlm_model=mlm_model, inputs=test_batch,
        tokenizer=preprocessor, num_samples=4,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    """Main entry point for CliffordNetEmbedding MLM pre-training."""
    parser = argparse.ArgumentParser(
        description="CliffordNetEmbedding MLM Pre-training",
    )
    parser.add_argument("--gpu", type=int, default=None, help="GPU device index")
    parser.add_argument(
        "--variant", type=str, default="nano",
        choices=["nano", "mini", "base", "large", "xl"],
        help="CliffordNetEmbedding variant",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--max-samples", type=int, default=10000, help="Max training samples",
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=128, help="Max sequence length",
    )
    parser.add_argument(
        "--pooling-strategy", type=str, default="mean",
        choices=["mean", "cls", "max"], help="Encoder pooling strategy",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke mode: tiny config + tiny dataset for CI/sanity",
    )
    args = parser.parse_args()

    setup_gpu(gpu_id=args.gpu)

    config = TrainingConfig()
    config.variant = args.variant
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.max_samples = args.max_samples
    config.max_seq_length = args.max_seq_length
    config.pooling_strategy = args.pooling_strategy

    if args.smoke:
        logger.info("SMOKE MODE: tiny config")
        config.variant = "nano"
        config.max_samples = max(args.max_samples or 0, 64) if args.max_samples else 64
        config.num_epochs = 1
        config.batch_size = min(args.batch_size, 8)
        config.max_seq_length = 32

    logger.info(
        f"Config: variant={config.variant}, epochs={config.num_epochs}, "
        f"batch_size={config.batch_size}, lr={config.learning_rate}, "
        f"max_samples={config.max_samples}, "
        f"max_seq_length={config.max_seq_length}, "
        f"pooling_strategy={config.pooling_strategy}, "
        f"pad_token_id={config.pad_token_id}"
    )

    mlm_model, history = train_cliffordnet_embedding_mlm(config)

    if not args.smoke:
        preprocessor = create_tokenizer(
            config.encoding_name, config.max_seq_length,
            config.cls_token_id, config.sep_token_id,
            config.pad_token_id, config.mask_token_id,
        )
        evaluate_model(mlm_model, preprocessor)

    logger.info(
        "Pre-training complete! Encoder saved under "
        f"{config.save_dir}/.../pretrained_cliffordnet_embedding_encoder_best.keras"
    )


if __name__ == "__main__":
    main()
