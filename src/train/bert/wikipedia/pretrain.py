"""
BERT Pre-training on Original Datasets (Wikipedia + BookCorpus).

This script performs large-scale pre-training of the BERT foundation model
using the original data sources described in the BERT paper:
1. English Wikipedia (approx. 2,500M words)
2. BookCorpus (approx. 800M words)

Features:
- Uses Hugging Face `datasets` in streaming mode to handle TB-scale text.
- Implements distributed training (MirroredStrategy) for multi-GPU setups.
- Uses Mixed Precision (AMP) for faster training and lower memory usage.
- Implements Masked Language Modeling (MLM).
"""

import os
import keras
import argparse
import numpy as np
import tensorflow as tf
import datasets  # Hugging Face datasets library
from typing import Dict, Tuple, Generator, Optional

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.models.bert import BERT
from dl_techniques.models.masked_language_model import MaskedLanguageModel
from dl_techniques.utils.logger import logger
from dl_techniques.utils.tokenizer import TiktokenPreprocessor
from dl_techniques.optimization import learning_rate_schedule_builder


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

class PretrainConfig:
    """Configuration for Full BERT Pre-training."""

    # Model Architecture (BERT-Base)
    bert_variant: str = "base"
    vocab_size: int = 100277
    max_seq_length: int = 512  # Standard BERT length

    # Tokenizer settings (Tiktoken cl100k_base)
    encoding_name: str = "cl100k_base"
    cls_token_id: int = 100264
    sep_token_id: int = 100265
    pad_token_id: int = 100266
    mask_token_id: int = 100267

    # Training Hyperparameters
    # Note: Original BERT trained for 1M steps with batch size 256
    global_batch_size: int = 32  # Adjust based on GPU VRAM (e.g., 256 for TPU/Cluster)
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 10000
    total_steps: int = 100000  # Scaled down for demonstration; use 1,000,000 for full convergence

    # MLM Settings
    mask_ratio: float = 0.15

    # System
    use_mixed_precision: bool = True
    save_dir: str = "results/bert_wiki_books_pretrain"
    log_freq: int = 100
    save_freq: int = 5000

    # Data Source Config
    # Loading is always streaming (`datasets.load_dataset(..., streaming=True)`);
    # there is no local-arrow-file branch, so there is no knob to select one.
    wikipedia_date: str = "20220301.en"


# ---------------------------------------------------------------------
# Data Pipeline (Wikipedia + BookCorpus)
# ---------------------------------------------------------------------

def create_tokenizer(config: PretrainConfig) -> TiktokenPreprocessor:
    return TiktokenPreprocessor(
        encoding_name=config.encoding_name,
        max_length=config.max_seq_length,
        cls_token_id=config.cls_token_id,
        sep_token_id=config.sep_token_id,
        pad_token_id=config.pad_token_id,
        mask_token_id=config.mask_token_id,
        truncation=True,
        padding='max_length',
    )


def get_dataset_generator(config: PretrainConfig) -> Generator[str, None, None]:
    """
    Creates a generator yielding text strings from Wikipedia and BookCorpus.
    Uses streaming to avoid downloading the entire 20GB+ dataset at once.
    """
    logger.info("Initializing Hugging Face Datasets streams...")

    # Load Wikipedia (Streaming)
    # We filter out headers and short articles
    wiki_ds = datasets.load_dataset(
        "wikipedia",
        config.wikipedia_date,
        split="train",
        streaming=True,
        trust_remote_code=True
    )

    # Load BookCorpus (Streaming)
    # Note: The original 'bookcorpus' on HF is often restricted.
    # 'bookcorpus/bookcorpus' is the standard open config.
    books_ds = datasets.load_dataset(
        "bookcorpus",
        split="train",
        streaming=True,
        trust_remote_code=True
    )

    # Interleave datasets (50% probability from each)
    # Original BERT used 2.5B words Wiki + 0.8B words Books.
    # We simply alternate here for simplicity.
    interleaved = datasets.interleave_datasets([wiki_ds, books_ds], probabilities=[0.5, 0.5])

    iter_ds = iter(interleaved)

    while True:
        try:
            item = next(iter_ds)
            text = item.get("text", "")

            # Basic cleaning: skip empty or very short lines
            if len(text) < 100:
                continue

            yield text
        except StopIteration:
            logger.info("Dataset iteration finished.")
            break
        except Exception as e:
            logger.warning(f"Error reading dataset stream: {e}")
            continue


def create_tf_dataset(
        config: PretrainConfig,
        preprocessor: TiktokenPreprocessor
) -> tf.data.Dataset:
    """Creates a highly optimized tf.data pipeline."""

    # 1. Create dataset from generator
    dataset = tf.data.Dataset.from_generator(
        lambda: get_dataset_generator(config),
        output_signature=tf.TensorSpec(shape=(), dtype=tf.string)
    )

    # 2. Tokenization Logic (running in graph mode)
    def tokenize_fn(text_tensor):
        # We need numpy for tiktoken, so we use py_function
        def _py_tokenize(text_bytes):
            text = text_bytes.decode('utf-8')
            enc = preprocessor(text, return_tensors='np')
            return (
                enc['input_ids'][0],
                enc['attention_mask'][0],
                enc['token_type_ids'][0]
            )

        input_ids, attn_mask, type_ids = tf.py_function(
            _py_tokenize,
            [text_tensor],
            [tf.int32, tf.int32, tf.int32]
        )

        # Set shapes explicitly for TF graph compilation
        input_ids.set_shape([config.max_seq_length])
        attn_mask.set_shape([config.max_seq_length])
        type_ids.set_shape([config.max_seq_length])

        return {
            'input_ids': input_ids,
            'attention_mask': attn_mask,
            'token_type_ids': type_ids
        }

    # 3. Pipeline optimization
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    dataset = dataset.with_options(options)
    # Parallel processing is crucial here due to Python tokenizer overhead
    dataset = dataset.map(tokenize_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(config.global_batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# ---------------------------------------------------------------------
# Training Setup
# ---------------------------------------------------------------------

def parse_arguments(argv: Optional[list] = None) -> argparse.Namespace:
    """Parse the CLI for Wikipedia+BookCorpus BERT pre-training.

    Kept deliberately small: one flag per `PretrainConfig` field it overrides,
    no invented knobs. `--gpu` is interface parity with every other trainer --
    `setup_gpu`'s `set_memory_growth` is a known repo-wide no-op, so device
    selection is still the `CUDA_VISIBLE_DEVICES=N` shell prefix.
    """
    parser = argparse.ArgumentParser(
        description="BERT MLM Pre-training on Wikipedia + BookCorpus"
    )
    parser.add_argument('--gpu', type=int, default=None, help='GPU device index')
    parser.add_argument('--variant', type=str, default=PretrainConfig.bert_variant,
                        help='BERT variant')
    parser.add_argument('--batch-size', type=int, default=PretrainConfig.global_batch_size,
                        help='Global batch size')
    parser.add_argument('--total-steps', type=int, default=PretrainConfig.total_steps,
                        help='Total training steps')
    parser.add_argument('--learning-rate', type=float, default=PretrainConfig.learning_rate,
                        help='Peak learning rate')
    return parser.parse_args(argv)


def main(argv: Optional[list] = None):
    # MUST be first: `--help` has to exit before MirroredStrategy, before the
    # dataset build (a live streaming HF Hub fetch) and before model
    # construction. Anything moved above this line makes `--help` expensive
    # again -- guarded by tests/test_train/test_bert_wikipedia/.
    args = parse_arguments(argv)

    # DECISION plan-2026-08-12T201216-50fc0975/D-006
    # `setup_gpu` is imported HERE, not at module scope. Do NOT hoist it back
    # up: `train/common/__init__.py` re-exports `image_text.py`, whose
    # module-level `tf.constant` (image_text.py:53) initializes TF's eager
    # context and ALLOCATES a GPU device at import time. A top-level
    # `from train.common import setup_gpu` therefore makes `--help` allocate a
    # GPU before argparse ever runs, defeating the point of this whole change.
    # Measured: `import train.common` -> 1 "Created device" line; `import
    # keras, tensorflow`, `dl_techniques.models.bert`, `datasets` and
    # `tensorflow_datasets` -> 0. See decisions.md D-006.
    from train.common import setup_gpu

    setup_gpu(gpu_id=args.gpu)

    config = PretrainConfig()
    config.bert_variant = args.variant
    config.global_batch_size = args.batch_size
    config.total_steps = args.total_steps
    config.learning_rate = args.learning_rate

    # 1. Hardware Strategy
    try:
        # Use all available GPUs
        strategy = tf.distribute.MirroredStrategy()
        logger.info(f"Training on {strategy.num_replicas_in_sync} GPUs.")
    except Exception:
        strategy = tf.distribute.get_strategy()  # Default
        logger.info("Training on default strategy (CPU/Single GPU).")

    # 2. Mixed Precision
    if config.use_mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")
        logger.info("Mixed precision (float16) enabled.")

    # 3. Data Pipeline
    logger.info("Building data pipeline for Wikipedia + BookCorpus...")
    tokenizer = create_tokenizer(config)
    dataset = create_tf_dataset(config, tokenizer)

    # 4. Model Initialization (Inside Strategy Scope)
    with strategy.scope():
        # Base Encoder
        # `BERT.__init__` has no `dropout_rate` parameter; it takes the two
        # separate probabilities below (both already default to 0.1). Passing
        # `dropout_rate=` raised `ValueError: Unrecognized keyword arguments`.
        # Call shape matches the working sibling `train/bert/pretrain.py:85-86`.
        bert_encoder = BERT.from_variant(
            variant=config.bert_variant,
            vocab_size=config.vocab_size,
            max_position_embeddings=config.max_seq_length,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )

        # MLM Wrapper
        mlm_model = MaskedLanguageModel(
            encoder=bert_encoder,
            vocab_size=config.vocab_size,
            mask_ratio=config.mask_ratio,
            mask_token_id=config.mask_token_id,
            special_token_ids=[
                config.cls_token_id, config.sep_token_id,
                config.pad_token_id, config.mask_token_id
            ]
        )

        # Optimization
        # Linear warmup then cosine decay
        # NOTE: `alpha` and `warmup_start_lr` are passed explicitly because the
        # builder's defaults (1e-4 and 1e-8) differ from the Keras/local defaults
        # this replaced (0.0 and 0.0). Omitting either changes the LR curve.
        lr_schedule = learning_rate_schedule_builder({
            "type": "cosine_decay",
            "learning_rate": config.learning_rate,
            "decay_steps": config.total_steps - config.warmup_steps,
            "alpha": 0.0,
            "warmup_steps": config.warmup_steps,
            "warmup_start_lr": 0.0,
        })

        optimizer = keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=config.weight_decay,
            clipnorm=1.0,
            jit_compile=True  # XLA Compilation for speed
        )

        # Mixed precision loss scaling is handled automatically by Keras in modern TF
        mlm_model.compile(optimizer=optimizer)

    # 5. Callbacks
    os.makedirs(config.save_dir, exist_ok=True)
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config.save_dir, "ckpt_{epoch}_{loss:.4f}.keras"),
            save_freq=config.save_freq
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(config.save_dir, "logs"),
            update_freq=config.log_freq
        ),
        # BackupAndRestore handles preemption nicely
        keras.callbacks.BackupAndRestore(
            backup_dir=os.path.join(config.save_dir, "backup")
        )
    ]

    # 6. Training Loop
    logger.info(f"Starting pre-training for {config.total_steps} steps...")

    # Since we use an infinite generator, we must specify steps_per_epoch
    # Note: We treat 'epochs' as just chunks of training steps here.
    mlm_model.fit(
        dataset,
        epochs=1,  # One giant epoch or split it up
        steps_per_epoch=config.total_steps,
        callbacks=callbacks
    )

    # 7. Save Final
    save_path = os.path.join(config.save_dir, "bert_wiki_books_final.keras")
    logger.info(f"Saving final model to {save_path}")
    mlm_model.encoder.save(save_path)
    logger.info("Pre-training complete.")


if __name__ == "__main__":
    main()