"""
BERT Fine-tuning on GLUE (SST-2).

This script fine-tunes a pre-trained BERT encoder on the Stanford Sentiment 
Treebank (SST-2) task from the GLUE benchmark. This is the standard 
"Hello World" for BERT evaluation.

Dataset: GLUE/SST-2 (loaded via tensorflow_datasets)
"""

import os
import keras
import argparse
import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Dict, Optional

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from train.common.nlp import build_headed_model_from_encoder, sentiment_task_config

from dl_techniques.models.language.bert import BERT
from dl_techniques.utils.logger import logger
from dl_techniques.utils.tokenizer import TiktokenPreprocessor


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

class FinetuneConfig:
    # Model Paths
    pretrained_encoder_path: str = "results/bert_wiki_books_pretrain/bert_wiki_books_final.keras"
    save_dir: str = "results/bert_glue_sst2"

    # Task Config (SST-2 is binary sentiment)
    task_name: str = "sst2"
    num_classes: int = 2

    # Tokenizer (Must match pre-training)
    encoding_name: str = "cl100k_base"
    max_seq_length: int = 128  # SST-2 sentences are short

    # Training Params
    batch_size: int = 32
    learning_rate: float = 3e-5  # Standard BERT fine-tuning rate
    epochs: int = 3  # Standard BERT fine-tuning epochs
    weight_decay: float = 0.01

    # Seeds Python, NumPy and Keras (train.common.set_seeds)
    seed: int = 42


# ---------------------------------------------------------------------
# Data Pipeline
# ---------------------------------------------------------------------

def create_tokenizer(config: FinetuneConfig) -> TiktokenPreprocessor:
    return TiktokenPreprocessor(
        encoding_name=config.encoding_name,
        max_length=config.max_seq_length,
        cls_token_id=100264,
        sep_token_id=100265,
        pad_token_id=100266,
        truncation=True,
        padding='max_length',
    )


def preprocess_glue(dataset: tf.data.Dataset, tokenizer: TiktokenPreprocessor, config: FinetuneConfig):
    """Tokenizes GLUE/SST-2 data.

    TFDS' GLUE builder has no supervised structure, so its elements are DICTS,
    not ``(text, label)`` pairs -- ``tfds.load(..., as_supervised=True)`` raises
    ``ValueError: as_supervised=True but glue does not support a supervised
    structure``. The feature names read below are the builder's own
    (``glue/sst2`` 2.0.0: ``sentence`` -> string, ``label`` -> int64 ClassLabel,
    plus an unused ``idx``).
    """

    def _tokenize(text, label):
        text_str = text.numpy().decode('utf-8')
        enc = tokenizer(text_str, return_tensors='np')
        return (
            enc['input_ids'][0],
            enc['attention_mask'][0],
            enc['token_type_ids'][0],
            label
        )

    def _wrapper(example: Dict[str, tf.Tensor]):
        input_ids, attn_mask, type_ids, label = tf.py_function(
            _tokenize,
            [example['sentence'], example['label']],
            [tf.int32, tf.int32, tf.int32, tf.int64]
        )

        # Structure for Keras Model
        inputs = {
            'input_ids': tf.ensure_shape(input_ids, [config.max_seq_length]),
            'attention_mask': tf.ensure_shape(attn_mask, [config.max_seq_length]),
            'token_type_ids': tf.ensure_shape(type_ids, [config.max_seq_length]),
        }
        return inputs, label

    dataset = dataset.map(_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(config.batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

def create_sst2_model(config: FinetuneConfig) -> keras.Model:
    """Rebuild the pretrained BERT encoder with a sentiment head (``include_top=True``).

    The architecture is read from the encoder checkpoint; the backbone loads
    completely or this RAISES. There is deliberately no fallback to a random
    encoder: a pretrained-encoder fine-tune that quietly starts from scratch
    reports a number that means nothing. See
    ``train.common.nlp.build_headed_model_from_encoder``.
    """
    logger.info(f"Loading Pretrained Encoder from {config.pretrained_encoder_path}...")
    return build_headed_model_from_encoder(
        BERT, config.pretrained_encoder_path, sentiment_task_config(config.num_classes),
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_arguments(argv: Optional[list] = None) -> argparse.Namespace:
    """Parse the CLI for GLUE/SST-2 fine-tuning.

    One flag per `FinetuneConfig` field it overrides; no invented knobs.
    """
    parser = argparse.ArgumentParser(description="BERT fine-tuning on GLUE/SST-2")
    parser.add_argument('--gpu', type=int, default=None, help='GPU device index')
    parser.add_argument('--batch-size', type=int, default=FinetuneConfig.batch_size,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=FinetuneConfig.epochs,
                        help='Fine-tuning epochs')
    parser.add_argument('--learning-rate', type=float, default=FinetuneConfig.learning_rate,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=FinetuneConfig.seed,
                        help='Seed for Python, NumPy and Keras (train.common.set_seeds)')
    return parser.parse_args(argv)


def main(argv: Optional[list] = None):
    # MUST be first: `--help` has to exit before `tfds.load` (which materializes
    # or downloads GLUE) and before any model construction.
    args = parse_arguments(argv)

    # DECISION plan-2026-08-12T201216-50fc0975/D-006
    # Function-scope import on purpose -- do NOT hoist. `train.common`'s
    # package init allocates a GPU device at import time (see the fuller note
    # in `pretrain.py`, and decisions.md D-006).
    #
    # SUPERSEDED 2026-08-13 by plan-2026-08-13T045759-fde437ba/D-003: that root
    # cause is FIXED (`image_text.py`'s constants are plain lists; `import
    # train.common` emits 0 `Created device` lines). This deferred import is
    # now harmless, not load-bearing. Kept for the history.
    from train.common import setup_gpu, set_seeds

    setup_gpu(gpu_id=args.gpu)

    config = FinetuneConfig()
    config.seed = args.seed
    set_seeds(config.seed)
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.learning_rate = args.learning_rate

    os.makedirs(config.save_dir, exist_ok=True)

    logger.info("Loading GLUE/SST-2 dataset...")
    # TFDS handles downloading the GLUE benchmark automatically
    # NOTE: no `as_supervised=True` -- GLUE has no supervised structure and TFDS
    # raises on it. `preprocess_glue` consumes the dict elements instead.
    data, _ = tfds.load(f'glue/{config.task_name}', with_info=True)

    tokenizer = create_tokenizer(config)
    train_ds = preprocess_glue(data['train'], tokenizer, config)
    val_ds = preprocess_glue(data['validation'], tokenizer, config)

    model = create_sst2_model(config)

    # Compile
    optimizer = keras.optimizers.AdamW(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # Train
    logger.info("Starting Fine-tuning on GLUE/SST-2...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                os.path.join(config.save_dir, "best_glue_model.keras"),
                save_best_only=True,
                monitor="val_accuracy"
            ),
            keras.callbacks.TensorBoard(os.path.join(config.save_dir, "logs"))
        ]
    )

    logger.info("Evaluation on validation set:")
    model.evaluate(val_ds)


if __name__ == "__main__":
    main()