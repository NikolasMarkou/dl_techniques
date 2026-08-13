"""
English-Only BERT Pre-training (Wikipedia + BookCorpus).

This script performs large-scale pre-training of the BERT foundation model
on the original English datasets. It includes explicit content filtering
to reject non-English text chunks that might appear in the raw streams.

Datasets:
1. English Wikipedia (20220301.en)
2. BookCorpus (English books)
"""

import os
import keras
import argparse
import tensorflow as tf
import datasets  # Hugging Face datasets
from typing import Generator, Optional

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
    """Configuration for English BERT Pre-training."""

    # Model
    bert_variant: str = "base"
    vocab_size: int = 100277
    max_seq_length: int = 512

    # Data Settings
    # Strictly use English Wikipedia
    wikipedia_id: str = "20220301.en"

    # Filter strictness: Max 10% non-ASCII characters allowed per sequence
    # This filters out CJK, Cyrillic, Arabic, etc., while allowing some accents.
    max_non_ascii_ratio: float = 0.1

    # Training Params
    global_batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 10000
    total_steps: int = 100000
    mask_ratio: float = 0.15

    # System
    save_dir: str = "results/bert_eng_pretrain"
    log_freq: int = 100
    save_freq: int = 5000

    # Tokenizer IDs (Tiktoken cl100k_base)
    encoding_name: str = "cl100k_base"
    cls_token_id: int = 100264
    sep_token_id: int = 100265
    pad_token_id: int = 100266
    mask_token_id: int = 100267


# ---------------------------------------------------------------------
# Data Pipeline & Filtering
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


def is_likely_english(text: str, max_non_ascii_ratio: float = 0.1) -> bool:
    """
    Fast heuristic to check if text is primarily English.

    We rely on ASCII density because pure English text is almost entirely ASCII.
    Foreign scripts (Chinese, Russian, etc.) will have very high non-ASCII counts.
    """
    if not text or len(text) == 0:
        return False

    # Optimization: If it encodes to ASCII without error, it's definitely English/Western
    try:
        text.encode('ascii')
        return True
    except UnicodeEncodeError:
        pass

    # Count non-ASCII characters
    non_ascii_count = sum(1 for c in text if ord(c) > 127)
    ratio = non_ascii_count / len(text)

    return ratio <= max_non_ascii_ratio


def get_english_dataset_generator(config: PretrainConfig) -> Generator[str, None, None]:
    """
    Yields English-only text from Wikipedia and BookCorpus streams.
    """
    logger.info("Initializing English Data Streams...")

    # 1. English Wikipedia
    logger.info(f"Loading Wikipedia ({config.wikipedia_id})...")
    wiki_ds = datasets.load_dataset(
        "wikipedia",
        config.wikipedia_id,
        split="train",
        streaming=True,
        trust_remote_code=True
    )

    # 2. BookCorpus (Generally English, but we will filter anyway)
    logger.info("Loading BookCorpus...")
    books_ds = datasets.load_dataset(
        "bookcorpus",
        split="train",
        streaming=True,
        trust_remote_code=True
    )

    # Interleave: 50% Wiki, 50% Books
    interleaved = datasets.interleave_datasets([wiki_ds, books_ds], probabilities=[0.5, 0.5])

    iter_ds = iter(interleaved)

    rejected_count = 0
    accepted_count = 0

    while True:
        try:
            item = next(iter_ds)
            text = item.get("text", "")

            # Filter 1: Length check (skip snippets too short for context)
            if len(text) < 100:
                continue

            # Filter 2: Strict English Heuristic
            if not is_likely_english(text, config.max_non_ascii_ratio):
                rejected_count += 1
                if rejected_count % 1000 == 0:
                    logger.debug(f"Filtered {rejected_count} non-English/dirty samples so far.")
                continue

            accepted_count += 1
            yield text

        except StopIteration:
            break
        except Exception as e:
            logger.warning(f"Stream error: {e}")
            continue


def create_tf_dataset(config: PretrainConfig, preprocessor: TiktokenPreprocessor) -> tf.data.Dataset:
    # Generator sets output type to string
    dataset = tf.data.Dataset.from_generator(
        lambda: get_english_dataset_generator(config),
        output_signature=tf.TensorSpec(shape=(), dtype=tf.string)
    )

    # Tokenization Wrapper
    def tokenize_fn(text_tensor):
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

        # Explicit shape setting for Graph mode
        input_ids.set_shape([config.max_seq_length])
        attn_mask.set_shape([config.max_seq_length])
        type_ids.set_shape([config.max_seq_length])

        return {
            'input_ids': input_ids,
            'attention_mask': attn_mask,
            'token_type_ids': type_ids
        }

    # Optimization
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    dataset = dataset.with_options(options)

    dataset = dataset.map(tokenize_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(config.global_batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# ---------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------

def parse_arguments(argv: Optional[list] = None) -> argparse.Namespace:
    """Parse the CLI for English-only BERT pre-training.

    Same minimal shape as the sibling `pretrain.py`, plus the one knob this
    fork actually adds: `--max-non-ascii-ratio`.
    """
    parser = argparse.ArgumentParser(
        description="English-only BERT MLM Pre-training on Wikipedia + BookCorpus"
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
    parser.add_argument('--max-non-ascii-ratio', type=float,
                        default=PretrainConfig.max_non_ascii_ratio,
                        help='Max fraction of non-ASCII characters allowed per sample')
    return parser.parse_args(argv)


def main(argv: Optional[list] = None):
    # MUST be first -- see the note in the sibling `pretrain.py`: `--help` has
    # to exit before MirroredStrategy, the streaming dataset build and model
    # construction.
    args = parse_arguments(argv)

    # DECISION plan-2026-08-12T201216-50fc0975/D-006
    # Function-scope import on purpose -- do NOT hoist. `train.common`'s
    # package init allocates a GPU device at import time (see the fuller note
    # in the sibling `pretrain.py`, and decisions.md D-006).
    #
    # SUPERSEDED 2026-08-13 by plan-2026-08-13T045759-fde437ba/D-003: that root
    # cause is FIXED (`image_text.py`'s constants are plain lists; `import
    # train.common` emits 0 `Created device` lines). This deferred import is
    # now harmless, not load-bearing. Kept for the history.
    from train.common import setup_gpu

    setup_gpu(gpu_id=args.gpu)

    config = PretrainConfig()
    config.bert_variant = args.variant
    config.global_batch_size = args.batch_size
    config.total_steps = args.total_steps
    config.learning_rate = args.learning_rate
    config.max_non_ascii_ratio = args.max_non_ascii_ratio

    # Strategy Setup
    try:
        strategy = tf.distribute.MirroredStrategy()
        logger.info(f"Using MirroredStrategy on {strategy.num_replicas_in_sync} devices.")
    except Exception:
        strategy = tf.distribute.get_strategy()
        logger.info("Using default strategy.")

    # Data Setup
    tokenizer = create_tokenizer(config)
    dataset = create_tf_dataset(config, tokenizer)

    # Model Setup
    with strategy.scope():
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

        # DECISION plan-2026-08-13T045759-fde437ba/D-004
        # Do NOT pass `jit_compile=` to a Keras 3 optimizer constructor, and do
        # NOT "fix" that by moving it to `mlm_model.compile(jit_compile=True)`.
        # This line used to read `jit_compile=True`; Keras 3 optimizers have no
        # such parameter, so it raised `ValueError: Argument(s) not recognized`
        # on an unconditional path and this script crashed at model compilation
        # on EVERY run. XLA also cannot compile this step under a distribution
        # strategy at all (`CollectiveGatherV2` has no XLA_GPU_JIT kernel; under
        # fp16 `LossScaleOptimizer`'s `Cond` additionally trips `merge_call`),
        # so moving the keyword to `compile()` would only relocate the crash.
        # Full measurement and the rejected `--jit-compile` alternative are
        # recorded once, at the sibling site in `pretrain.py`. Read that note
        # before changing either file.
        optimizer = keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=config.weight_decay,
            clipnorm=1.0,
        )

        mlm_model.compile(optimizer=optimizer)

    # Training
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
        keras.callbacks.BackupAndRestore(
            backup_dir=os.path.join(config.save_dir, "backup")
        )
    ]

    logger.info("Starting English-only BERT training...")
    mlm_model.fit(
        dataset,
        epochs=1,
        steps_per_epoch=config.total_steps,
        callbacks=callbacks
    )

    # Save
    save_path = os.path.join(config.save_dir, "bert_eng_final.keras")
    mlm_model.encoder.save(save_path)
    logger.info(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()