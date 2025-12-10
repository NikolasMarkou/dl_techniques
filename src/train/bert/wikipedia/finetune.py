"""
BERT Fine-tuning on GLUE (SST-2).

This script fine-tunes a pre-trained BERT encoder on the Stanford Sentiment 
Treebank (SST-2) task from the GLUE benchmark. This is the standard 
"Hello World" for BERT evaluation.

Dataset: GLUE/SST-2 (loaded via tensorflow_datasets)
"""

import os
import keras
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from typing import Dict, Tuple

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.models.bert import BERT
from dl_techniques.utils.logger import logger
from dl_techniques.utils.tokenizer import TiktokenPreprocessor
from dl_techniques.layers.nlp_heads import NLPTaskConfig, NLPTaskType, create_nlp_head


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
    """Tokenizes GLUE/SST-2 data."""

    def _tokenize(text, label):
        text_str = text.numpy().decode('utf-8')
        enc = tokenizer(text_str, return_tensors='np')
        return (
            enc['input_ids'][0],
            enc['attention_mask'][0],
            enc['token_type_ids'][0],
            label
        )

    def _wrapper(text, label):
        input_ids, attn_mask, type_ids, label = tf.py_function(
            _tokenize,
            [text, label],
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
# Main
# ---------------------------------------------------------------------

def main():
    config = FinetuneConfig()
    os.makedirs(config.save_dir, exist_ok=True)

    logger.info("Loading GLUE/SST-2 dataset...")
    # TFDS handles downloading the GLUE benchmark automatically
    data, info = tfds.load(f'glue/{config.task_name}', with_info=True, as_supervised=True)

    tokenizer = create_tokenizer(config)
    train_ds = preprocess_glue(data['train'], tokenizer, config)
    val_ds = preprocess_glue(data['validation'], tokenizer, config)

    logger.info(f"Loading Pretrained Encoder from {config.pretrained_encoder_path}...")
    try:
        bert_encoder = keras.models.load_model(
            config.pretrained_encoder_path,
            custom_objects={"BERT": BERT}
        )
    except Exception as e:
        logger.error(f"Could not load pretrained encoder: {e}")
        logger.warning("Initializing random encoder for demonstration.")
        bert_encoder = BERT.from_variant("base", vocab_size=100277)

    # Build Classifier
    task_config = NLPTaskConfig(
        name="sst2_head",
        task_type=NLPTaskType.SENTIMENT_ANALYSIS,
        num_classes=config.num_classes
    )
    head = create_nlp_head(task_config, input_dim=bert_encoder.hidden_size)

    # Assemble End-to-End Model
    inputs = {
        "input_ids": keras.Input(shape=(config.max_seq_length,), dtype="int32", name="input_ids"),
        "attention_mask": keras.Input(shape=(config.max_seq_length,), dtype="int32", name="attention_mask"),
        "token_type_ids": keras.Input(shape=(config.max_seq_length,), dtype="int32", name="token_type_ids"),
    }

    # Forward pass
    enc_out = bert_encoder(inputs)
    # CLS token is at index 0
    cls_embedding = enc_out["last_hidden_state"][:, 0, :]

    # Head pass
    # Note: Factory heads expect a dict or tensor depending on impl.
    # Providing dict for compatibility with BaseNLPHead logic.
    head_out = head({"hidden_states": cls_embedding})

    model = keras.Model(inputs, head_out['logits'])

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