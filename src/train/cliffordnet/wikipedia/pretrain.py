"""CliffordNetEmbedding MLM Pre-training on local Wikipedia (Pattern-3 NLP).

Mirrors :mod:`train.bert.wikipedia.pretrain` for the data/callback/mega-epoch
shape and :mod:`train.cliffordnet.train_embeddings` for the model/optimizer/
argparse shape. Reads Wikipedia from the local Arrow cache via
:func:`dl_techniques.datasets.nlp.load_wikipedia_train_val` (NOT HF streaming).

Usage:
    MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=0 \\
        .venv/bin/python -m train.cliffordnet.wikipedia.pretrain --smoke

    MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=0 \\
        .venv/bin/python -m train.cliffordnet.wikipedia.pretrain \\
            --variant nano --batch-size 32 --max-seq-length 512 \\
            --total-steps 50000 --warmup-steps 5000

Single GPU only. Never run two of these in parallel.
"""

import argparse
import os

import keras
import tensorflow as tf

from train.common import setup_gpu
from train.common.nlp import create_tokenizer, preprocess_mlm_dataset

from dl_techniques.datasets.nlp import load_wikipedia_train_val
from dl_techniques.models.cliffordnet import CliffordNetEmbedding
from dl_techniques.models.masked_language_model import MaskedLanguageModel
from dl_techniques.optimization.warmup_schedule import WarmupSchedule
from dl_techniques.utils.logger import logger
from dl_techniques.utils.tokenizer import TiktokenPreprocessor


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


class PretrainConfig:
    """Configuration for CliffordNetEmbedding Wikipedia MLM pre-training."""

    # Model
    variant: str = "nano"
    vocab_size: int = 100277  # tiktoken cl100k_base
    max_seq_length: int = 512
    pooling_strategy: str = "mean"

    # Tokenizer (Tiktoken cl100k_base)
    encoding_name: str = "cl100k_base"
    cls_token_id: int = 100264
    sep_token_id: int = 100265
    pad_token_id: int = 100266
    mask_token_id: int = 100267

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 5000
    total_steps: int = 50000

    # Validation
    validation_steps: int = 50
    validation_freq: int = 1  # epochs; with 1 mega-epoch it runs once at end

    # MLM
    mask_ratio: float = 0.15
    random_token_ratio: float = 0.1
    unchanged_ratio: float = 0.1

    # System
    use_mixed_precision: bool = True
    save_dir: str = "results/cliffordnet_embedding_wiki_pretrain"
    log_freq: int = 100
    save_freq: int = 5000
    seed: int = 42

    # Data
    wikipedia_cache_dir: str = "/media/arxwn/data0_4tb/datasets/wikipedia"
    wikipedia_config_name: str = "20231101.en"
    min_article_length: int = 500
    val_fraction: float = 0.02
    max_val_samples: int = 5000
    max_train_samples: int = None  # type: ignore[assignment]
    num_shards: int = 4


# ---------------------------------------------------------------------
# Tokenizer / Model / Dataset
# ---------------------------------------------------------------------


def create_tokenizer_from_config(config: PretrainConfig) -> TiktokenPreprocessor:
    """Build the tiktoken preprocessor wired to the configured special-token IDs."""
    return create_tokenizer(
        encoding_name=config.encoding_name,
        max_length=config.max_seq_length,
        cls_token_id=config.cls_token_id,
        sep_token_id=config.sep_token_id,
        pad_token_id=config.pad_token_id,
        mask_token_id=config.mask_token_id,
    )


def create_mlm_model(config: PretrainConfig) -> MaskedLanguageModel:
    """Build CliffordNetEmbedding encoder + MaskedLanguageModel wrapper."""
    logger.info(
        f"Creating CliffordNetEmbedding-{config.variant.upper()} encoder..."
    )
    # NOTE: pad_token_id wired explicitly into the encoder ctor — tokenizer-aware
    # encoders need this for correct attention/pooling masking.
    encoder = CliffordNetEmbedding.from_variant(
        variant=config.variant,
        vocab_size=config.vocab_size,
        max_seq_length=config.max_seq_length,
        pooling_strategy=config.pooling_strategy,
        pad_token_id=config.pad_token_id,
        dropout_rate=0.1,
    )

    special_token_ids = [
        config.cls_token_id,
        config.sep_token_id,
        config.pad_token_id,
        config.mask_token_id,
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

    # Build to count params.
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


def build_datasets(config: PretrainConfig, preprocessor: TiktokenPreprocessor):
    """Load Wikipedia train/val + tokenize/batch for MLM.

    Returns
    -------
    (train_ds, val_ds, n_train, n_val)
    """
    train_ds, val_ds, n_train, n_val = load_wikipedia_train_val(
        cache_dir=config.wikipedia_cache_dir,
        config_name=config.wikipedia_config_name,
        min_article_length=config.min_article_length,
        val_fraction=config.val_fraction,
        max_val_samples=config.max_val_samples,
        max_train_samples=config.max_train_samples,
        seed=config.seed,
        return_counts=True,
        num_shards=config.num_shards,
    )
    logger.info(f"Wikipedia split: train={n_train:,} val={n_val:,}")

    train_ds = preprocess_mlm_dataset(
        train_ds, preprocessor, config.max_seq_length, config.batch_size,
    )
    val_ds = preprocess_mlm_dataset(
        val_ds, preprocessor, config.max_seq_length, config.batch_size,
    )
    # Val pipeline finite by default; repeat so validation_steps can pull
    # consistently across runs without StopIteration.
    val_ds = val_ds.repeat()
    return train_ds, val_ds, n_train, n_val


# ---------------------------------------------------------------------
# Optimizer / Callbacks
# ---------------------------------------------------------------------


def compile_model(mlm_model: MaskedLanguageModel, config: PretrainConfig) -> None:
    """Compile with AdamW + WarmupSchedule(CosineDecay)."""
    decay_steps = max(1, config.total_steps - config.warmup_steps)
    lr_schedule = WarmupSchedule(
        warmup_steps=config.warmup_steps,
        primary_schedule=keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=config.learning_rate,
            decay_steps=decay_steps,
        ),
        warmup_start_lr=0.0,
    )
    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=config.weight_decay,
        clipnorm=1.0,
    )
    mlm_model.compile(optimizer=optimizer)
    logger.info(
        f"Compiled: AdamW, peak_lr={config.learning_rate}, "
        f"wd={config.weight_decay}, warmup={config.warmup_steps}, "
        f"total_steps={config.total_steps}"
    )


def create_training_callbacks(config: PretrainConfig):
    """ModelCheckpoint by save_freq + TensorBoard + BackupAndRestore.

    Mirrors bert/wikipedia: NO EpochAnalyzer (mega-epoch = analyzer runs once
    after potentially hours of compute; not worth the overhead here).
    """
    os.makedirs(config.save_dir, exist_ok=True)
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(
                config.save_dir, "ckpt_step{epoch}_{loss:.4f}.keras",
            ),
            save_freq=config.save_freq,
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(config.save_dir, "logs"),
            update_freq=config.log_freq,
        ),
        keras.callbacks.BackupAndRestore(
            backup_dir=os.path.join(config.save_dir, "backup"),
        ),
    ]


# ---------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------


def train(config: PretrainConfig):
    """End-to-end training entry point.

    Returns the `keras.callbacks.History` from `fit`.
    """
    logger.info("=" * 60)
    logger.info("CliffordNetEmbedding Wikipedia MLM Pre-training")
    logger.info("=" * 60)

    tf.random.set_seed(config.seed)
    keras.utils.set_random_seed(config.seed)
    os.makedirs(config.save_dir, exist_ok=True)

    if config.use_mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")
        logger.info("Mixed precision (float16) enabled.")

    preprocessor = create_tokenizer_from_config(config)
    train_ds, val_ds, n_train, n_val = build_datasets(config, preprocessor)
    mlm_model = create_mlm_model(config)
    compile_model(mlm_model, config)
    callbacks = create_training_callbacks(config)

    logger.info(
        f"Starting fit: total_steps={config.total_steps}, "
        f"batch_size={config.batch_size}, "
        f"max_seq_length={config.max_seq_length}, "
        f"variant={config.variant}, mixed_fp16={config.use_mixed_precision}"
    )
    history = mlm_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=1,
        steps_per_epoch=config.total_steps,
        validation_steps=config.validation_steps,
        validation_freq=config.validation_freq,
        callbacks=callbacks,
        verbose=1,
    )
    logger.info("Training complete.")

    encoder_path = os.path.join(
        config.save_dir, "cliffordnet_embedding_wiki_final.keras",
    )
    logger.info(f"Saving encoder to {encoder_path}")
    mlm_model.encoder.save(encoder_path)
    logger.info("Encoder saved.")
    return history


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def _apply_smoke_overrides(config: PretrainConfig) -> None:
    """Tiny config for sanity validation. Mixed precision OFF (faster build)."""
    config.variant = "nano"
    config.max_seq_length = 128
    config.batch_size = 8
    config.warmup_steps = 40
    config.total_steps = 400
    config.validation_steps = 4
    config.validation_freq = 1
    config.use_mixed_precision = False
    config.save_dir = "results/cliffordnet_embedding_wiki_smoke"
    config.log_freq = 20
    config.save_freq = 200
    config.max_val_samples = 200
    config.max_train_samples = 4000  # ~4000 articles is plenty for 400 steps
    config.min_article_length = 500
    config.num_shards = 2
    logger.info("SMOKE MODE: tiny config + tiny dataset for sanity validation.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CliffordNetEmbedding Wikipedia MLM Pre-training",
    )
    parser.add_argument("--gpu", type=int, default=None, help="GPU device index")
    parser.add_argument(
        "--variant", type=str, default="nano",
        choices=["nano", "mini", "base", "large", "xl"],
        help="CliffordNetEmbedding variant",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument(
        "--pooling-strategy", type=str, default="mean",
        choices=["mean", "cls", "max"],
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=5000)
    parser.add_argument("--total-steps", type=int, default=50000)
    parser.add_argument("--validation-steps", type=int, default=50)
    parser.add_argument("--validation-freq", type=int, default=1)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--save-freq", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument(
        "--min-article-length", type=int, default=500,
        help="Filter Wikipedia articles shorter than this many characters.",
    )
    parser.add_argument(
        "--no-mixed-precision", action="store_true",
        help="Disable mixed_float16 global policy.",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Tiny config + tiny dataset for ~5 min sanity validation.",
    )
    args = parser.parse_args()

    setup_gpu(gpu_id=args.gpu)

    config = PretrainConfig()
    config.variant = args.variant
    config.batch_size = args.batch_size
    config.max_seq_length = args.max_seq_length
    config.pooling_strategy = args.pooling_strategy
    config.learning_rate = args.learning_rate
    config.weight_decay = args.weight_decay
    config.warmup_steps = args.warmup_steps
    config.total_steps = args.total_steps
    config.validation_steps = args.validation_steps
    config.validation_freq = args.validation_freq
    config.log_freq = args.log_freq
    config.save_freq = args.save_freq
    config.seed = args.seed
    config.num_shards = args.num_shards
    config.min_article_length = args.min_article_length
    if args.save_dir is not None:
        config.save_dir = args.save_dir
    if args.no_mixed_precision:
        config.use_mixed_precision = False

    if args.smoke:
        _apply_smoke_overrides(config)

    logger.info(
        f"Config: variant={config.variant} bs={config.batch_size} "
        f"seq={config.max_seq_length} lr={config.learning_rate} "
        f"warmup={config.warmup_steps} total={config.total_steps} "
        f"mixed_fp16={config.use_mixed_precision} save_dir={config.save_dir} "
        f"pad_token_id={config.pad_token_id}"
    )

    train(config)
    logger.info(
        f"Pre-training complete. Encoder at "
        f"{config.save_dir}/cliffordnet_embedding_wiki_final.keras"
    )


if __name__ == "__main__":
    main()
