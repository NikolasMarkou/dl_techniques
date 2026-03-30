"""FNet Pre-training Script with Masked Language Modeling.

Pre-trains an FNet encoder using MLM on a text dataset (IMDB reviews by default).
Saves both the full MLM model and the encoder separately for downstream fine-tuning.
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

from dl_techniques.models.fnet import FNet
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
    """Configuration for FNet MLM pre-training."""

    # Model
    fnet_variant: str = "tiny"
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
    save_dir: str = "results/fnet_pretrain"

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


def create_fnet_mlm_model(config: TrainingConfig) -> MaskedLanguageModel:
    """Create FNet encoder wrapped in MaskedLanguageModel."""
    logger.info(f"Creating FNet-{config.fnet_variant.upper()} encoder...")
    encoder = FNet.from_variant(
        variant=config.fnet_variant,
        vocab_size=config.vocab_size,
        max_position_embeddings=config.max_seq_length,
        hidden_dropout_prob=0.1,
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


def compile_model(mlm_model: MaskedLanguageModel, config: TrainingConfig, steps_per_epoch: int):
    """Compile MLM model with AdamW and warmup schedule."""
    lr_schedule = create_warmup_lr_schedule(
        config.learning_rate, config.num_epochs, steps_per_epoch, config.warmup_ratio,
    )
    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule, weight_decay=config.weight_decay, clipnorm=1.0,
    )
    mlm_model.compile(optimizer=optimizer)
    logger.info(f"Compiled: AdamW, peak_lr={config.learning_rate}, wd={config.weight_decay}")


def train_fnet_mlm(config: TrainingConfig) -> Tuple[MaskedLanguageModel, keras.callbacks.History]:
    """Run FNet MLM pre-training."""
    logger.info("=" * 60)
    logger.info("FNet MLM Pre-training with Tiktoken")
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

    steps_per_epoch = config.max_samples // config.batch_size if config.max_samples else 1000
    mlm_model = create_fnet_mlm_model(config)
    compile_model(mlm_model, config, steps_per_epoch)
    callbacks, results_dir = create_nlp_callbacks(
        model_name=f"FNet-{config.fnet_variant}",
        results_dir_prefix="fnet_pretrain",
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
    final_path = os.path.join(config.save_dir, "fnet_mlm_final_best.keras")
    mlm_model.save(final_path)
    encoder_path = os.path.join(config.save_dir, "pretrained_fnet_encoder_best.keras")
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
    """Main entry point for FNet MLM pre-training."""
    parser = argparse.ArgumentParser(description="FNet MLM Pre-training")
    parser.add_argument('--gpu', type=int, default=None, help='GPU device index')
    parser.add_argument('--variant', type=str, default='tiny', help='FNet variant')
    parser.add_argument('--epochs', type=int, default=3, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--max-samples', type=int, default=10000, help='Max training samples')
    args = parser.parse_args()

    setup_gpu(gpu_id=args.gpu)

    config = TrainingConfig()
    config.fnet_variant = args.variant
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.max_samples = args.max_samples

    logger.info(f"Config: variant={config.fnet_variant}, epochs={config.num_epochs}, "
                f"batch_size={config.batch_size}, lr={config.learning_rate}, "
                f"max_samples={config.max_samples}")

    mlm_model, history = train_fnet_mlm(config)

    preprocessor = create_tokenizer(
        config.encoding_name, config.max_seq_length,
        config.cls_token_id, config.sep_token_id,
        config.pad_token_id, config.mask_token_id,
    )
    evaluate_model(mlm_model, preprocessor)

    logger.info(f"Pre-training complete! Encoder: {config.save_dir}/pretrained_fnet_encoder_best.keras")


if __name__ == "__main__":
    main()
