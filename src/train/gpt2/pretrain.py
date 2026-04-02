"""GPT-2 Pre-training Script with Causal Language Modeling.

Pre-trains a GPT-2 decoder on a text dataset using next-token prediction
(causal LM). Supports both TFDS datasets (IMDB) and HuggingFace datasets
(Wikipedia, OpenWebText, etc.). Saves the trained model for downstream
fine-tuning or text generation.
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

from dl_techniques.datasets.nlp import load_wikipedia_dataset, load_hf_text_dataset
from dl_techniques.models.gpt2 import GPT2
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


class TrainingConfig:
    """Configuration for GPT-2 CLM pre-training."""

    # Model
    gpt2_variant: str = "tiny"
    vocab_size: int = 50261  # Tiktoken gpt2 (50257) + 4 special tokens
    max_seq_length: int = 128
    num_layers: int = 8  # Override tiny default (4) for better capacity
    num_heads: int = 8  # More heads for 8-layer model

    # Tokenizer (Tiktoken gpt2 — 50K vocab, half the embedding cost)
    encoding_name: str = "gpt2"
    cls_token_id: int = 50257
    sep_token_id: int = 50258
    pad_token_id: int = 50259
    mask_token_id: int = 50260

    # Training
    batch_size: int = 32
    num_epochs: int = 3
    learning_rate: float = 5e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

    # Paths
    save_dir: str = "results/gpt2_pretrain"

    # Data source: "huggingface" or "tfds"
    dataset_source: str = "huggingface"

    # TFDS settings (used when dataset_source="tfds")
    dataset_name: str = "imdb_reviews"
    max_samples: Optional[int] = 10000

    # HuggingFace settings (used when dataset_source="huggingface")
    hf_dataset_path: str = "wikimedia/wikipedia"
    hf_dataset_name: Optional[str] = "20231101.en"
    hf_text_field: str = "text"
    hf_cache_dir: str = "/media/arxwn/data0_4tb/datasets/wikipedia"
    min_article_length: int = 100
    streaming: bool = True
    steps_per_epoch: int = 5000
    val_samples: int = 2000
    val_skip_samples: int = 500000

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
        depth=config.num_layers,
        num_heads=config.num_heads,
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
    )

    # Build to count parameters
    import numpy as np
    dummy = np.random.randint(
        0, config.vocab_size, size=(1, config.max_seq_length - 1)
    ).astype("int32")
    _ = model(dummy, training=False)

    total_p = model.count_params()
    logger.info(f"GPT-2 model: {total_p:,} parameters")
    return model


# ---------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------


def _wrap_labels_for_dict_output(
    dataset: tf.data.Dataset,
) -> tf.data.Dataset:
    """Wrap (input_ids, labels, sample_weight) for dict-output model.

    GPT-2 model returns ``{"logits": ..., "last_hidden_state": ...}``,
    so Keras expects labels in dict format matching the output keys.
    The ``sample_weight`` masks PAD positions out of the loss.
    """
    return dataset.map(
        lambda x, y, w: (x, {"logits": y}, {"logits": w}),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def load_train_val_datasets(
    config: TrainingConfig,
    preprocessor,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Load and preprocess train/val datasets based on config."""
    if config.dataset_source == "tfds":
        train_ds, val_ds = _load_tfds_datasets(config, preprocessor)
    elif config.dataset_source == "huggingface":
        train_ds, val_ds = _load_hf_datasets(config, preprocessor)
    else:
        raise ValueError(
            f"Unknown dataset_source: {config.dataset_source!r}. "
            f"Use 'tfds' or 'huggingface'."
        )
    return _wrap_labels_for_dict_output(train_ds), _wrap_labels_for_dict_output(val_ds)


def _load_tfds_datasets(
    config: TrainingConfig,
    preprocessor,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Load train/val from TFDS (e.g. IMDB)."""
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
    return train_dataset, val_dataset


def _load_hf_datasets(
    config: TrainingConfig,
    preprocessor,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Load train/val from HuggingFace (e.g. Wikipedia)."""
    # Training set: streaming for large datasets
    if "wikipedia" in config.hf_dataset_path:
        train_raw = load_wikipedia_dataset(
            cache_dir=config.hf_cache_dir,
            config_name=config.hf_dataset_name,
            min_article_length=config.min_article_length,
            max_samples=config.max_samples if not config.streaming else None,
            streaming=config.streaming,
        )
    else:
        train_raw = load_hf_text_dataset(
            path=config.hf_dataset_path,
            name=config.hf_dataset_name,
            text_field=config.hf_text_field,
            cache_dir=config.hf_cache_dir,
            min_length=config.min_article_length,
            max_samples=config.max_samples if not config.streaming else None,
            streaming=config.streaming,
        )

    train_dataset = preprocess_clm_dataset(
        train_raw,
        preprocessor,
        config.max_seq_length,
        config.batch_size,
        streaming=config.streaming,
    )

    # Validation set: skip first N articles to avoid train/val overlap
    logger.info(
        f"Loading validation set: {config.val_samples} samples "
        f"(skip_samples={config.val_skip_samples})"
    )
    if "wikipedia" in config.hf_dataset_path:
        val_raw = load_wikipedia_dataset(
            cache_dir=config.hf_cache_dir,
            config_name=config.hf_dataset_name,
            min_article_length=config.min_article_length,
            max_samples=config.val_samples,
            skip_samples=config.val_skip_samples,
            streaming=True,
        )
    else:
        val_raw = load_hf_text_dataset(
            path=config.hf_dataset_path,
            name=config.hf_dataset_name,
            text_field=config.hf_text_field,
            cache_dir=config.hf_cache_dir,
            min_length=config.min_article_length,
            max_samples=config.val_samples,
            skip_samples=config.val_skip_samples,
            streaming=True,
        )

    val_dataset = preprocess_clm_dataset(
        val_raw,
        preprocessor,
        config.max_seq_length,
        config.batch_size,
        streaming=False,
    )
    return train_dataset, val_dataset


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
        loss={"logits": keras.losses.SparseCategoricalCrossentropy(from_logits=True)},
        metrics={"logits": ["accuracy"]},
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
    train_dataset, val_dataset = load_train_val_datasets(config, preprocessor)

    # Steps per epoch
    if config.dataset_source == "huggingface" and config.streaming:
        steps_per_epoch = config.steps_per_epoch
    elif config.max_samples:
        steps_per_epoch = config.max_samples // config.batch_size
    else:
        steps_per_epoch = 1000

    # Model
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
    logger.info(
        f"Starting training: source={config.dataset_source}, "
        f"steps_per_epoch={steps_per_epoch}"
    )
    fit_kwargs = dict(
        epochs=config.num_epochs,
        callbacks=callbacks,
        validation_data=val_dataset,
        verbose=1,
    )
    if config.dataset_source == "huggingface" and config.streaming:
        fit_kwargs["steps_per_epoch"] = steps_per_epoch

    history = model.fit(train_dataset, **fit_kwargs)
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
        default=None,
        help="Max training samples (None=unlimited for streaming)",
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

    # Dataset source
    parser.add_argument(
        "--dataset-source",
        type=str,
        default="huggingface",
        choices=["tfds", "huggingface"],
        help="Dataset source: 'tfds' for TensorFlow Datasets, "
             "'huggingface' for HuggingFace Hub",
    )

    # TFDS settings
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="imdb_reviews",
        help="TFDS dataset name (when --dataset-source=tfds)",
    )

    # HuggingFace settings
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default="wikimedia/wikipedia",
        help="HuggingFace dataset path (e.g. 'wikimedia/wikipedia', 'openwebtext')",
    )
    parser.add_argument(
        "--hf-config",
        type=str,
        default="20231101.en",
        help="HuggingFace dataset config name",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=str,
        default="/media/arxwn/data0_4tb/datasets/wikipedia",
        help="Local cache directory for HuggingFace datasets",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=True,
        help="Use streaming mode for HF datasets (default: True)",
    )
    parser.add_argument(
        "--no-streaming",
        dest="streaming",
        action="store_false",
        help="Disable streaming (downloads full dataset first)",
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=5000,
        help="Steps per epoch when streaming (default: 5000)",
    )
    parser.add_argument(
        "--val-samples",
        type=int,
        default=2000,
        help="Number of validation samples (default: 2000)",
    )
    args = parser.parse_args()

    setup_gpu(gpu_id=args.gpu)

    config = TrainingConfig()
    config.gpt2_variant = args.variant
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.max_seq_length = args.max_seq_length
    config.learning_rate = args.learning_rate
    config.dataset_source = args.dataset_source
    config.streaming = args.streaming
    config.steps_per_epoch = args.steps_per_epoch
    config.val_samples = args.val_samples

    if args.max_samples is not None:
        config.max_samples = args.max_samples

    # TFDS settings
    config.dataset_name = args.dataset_name

    # HuggingFace settings
    config.hf_dataset_path = args.hf_dataset
    config.hf_dataset_name = args.hf_config
    config.hf_cache_dir = args.hf_cache_dir

    logger.info(
        f"Config: variant={config.gpt2_variant}, epochs={config.num_epochs}, "
        f"batch_size={config.batch_size}, lr={config.learning_rate}, "
        f"source={config.dataset_source}, streaming={config.streaming}"
    )

    model, history = train_gpt2(config)
    logger.info(
        f"Pre-training complete! Model: {config.save_dir}/gpt2_final_best.keras"
    )


if __name__ == "__main__":
    main()
