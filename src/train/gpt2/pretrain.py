"""GPT-2 Pre-training Script with Causal Language Modeling.

Pre-trains a GPT-2 decoder on a text dataset using next-token prediction
(causal LM). Supports both TFDS datasets (IMDB) and HuggingFace datasets
(Wikipedia, OpenWebText, etc.). Saves the trained model for downstream
fine-tuning or text generation.
"""

import os
import keras
import argparse
import tensorflow as tf
from typing import Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

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
from dl_techniques.datasets.nlp import load_wikipedia_train_val
from dl_techniques.losses import MaskedCausalLMLoss, FocalCausalLMLoss
from dl_techniques.analyzer import ModelAnalyzer, AnalysisConfig


# ---------------------------------------------------------------------
# Step-based Checkpoint & Analysis Callback
# ---------------------------------------------------------------------


class StepCheckpointCallback(keras.callbacks.Callback):
    """Save model checkpoints and run analyzer at fixed step intervals.

    For large datasets where one epoch = 300K+ steps, epoch-level
    checkpoints are too infrequent. This callback saves the model and
    optionally runs weight/spectral analysis every ``save_every_steps``
    training steps.

    :param save_dir: Directory for checkpoint files.
    :param save_every_steps: Save checkpoint every N steps.
    :param analyze_every_steps: Run model analysis every N steps.
        Set to 0 to disable analysis. Defaults to ``save_every_steps``.
    :param model_name: Name used in analyzer output.
    """

    def __init__(
        self,
        save_dir: str,
        save_every_steps: int = 25000,
        analyze_every_steps: int = 25000,
        model_name: str = "gpt2",
    ):
        super().__init__()
        self.save_dir = save_dir
        self.save_every_steps = save_every_steps
        self.analyze_every_steps = analyze_every_steps
        self.model_name = model_name
        self._global_step = 0

        self._ckpt_dir = os.path.join(save_dir, "checkpoints")
        self._analysis_dir = os.path.join(save_dir, "step_analysis")
        os.makedirs(self._ckpt_dir, exist_ok=True)
        if analyze_every_steps > 0:
            os.makedirs(self._analysis_dir, exist_ok=True)

        self._analysis_config = AnalysisConfig(
            analyze_weights=True,
            analyze_spectral=True,
            analyze_calibration=False,
            analyze_information_flow=False,
            analyze_training_dynamics=False,
            verbose=False,
        )

        logger.info(
            f"StepCheckpointCallback: save every {save_every_steps} steps, "
            f"analyze every {analyze_every_steps} steps"
        )

    def on_train_batch_end(self, batch, logs=None):
        self._global_step += 1

        if self._global_step % self.save_every_steps == 0:
            self._save_checkpoint()

        if (
            self.analyze_every_steps > 0
            and self._global_step % self.analyze_every_steps == 0
        ):
            self._run_analysis()

    def _save_checkpoint(self):
        path = os.path.join(
            self._ckpt_dir, f"step_{self._global_step:07d}.keras"
        )
        self.model.save(path)
        logger.info(
            f"Checkpoint saved: {path} (step {self._global_step:,})"
        )

    def _run_analysis(self):
        step_dir = os.path.join(
            self._analysis_dir, f"step_{self._global_step:07d}"
        )
        try:
            analyzer = ModelAnalyzer(
                models={self.model_name: self.model},
                config=self._analysis_config,
                output_dir=step_dir,
            )
            _ = analyzer.analyze()
            logger.info(
                f"Step analysis complete: step {self._global_step:,}"
            )
        except Exception as e:
            logger.error(f"Step analysis failed at step {self._global_step}: {e}")

    def on_train_end(self, logs=None):
        # Always save final checkpoint
        path = os.path.join(self._ckpt_dir, "final.keras")
        self.model.save(path)
        logger.info(f"Final checkpoint saved: {path}")


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

    # Loss: "ce" (default) or "focal" (focal cross-entropy)
    loss_type: str = "ce"
    focal_gamma: float = 1.0
    label_smoothing: float = 0.0

    # Paths
    save_dir: str = "results/gpt2_pretrain"

    # Data source: "huggingface" or "tfds"
    dataset_source: str = "huggingface"

    # TFDS settings (used when dataset_source="tfds")
    dataset_name: str = "imdb_reviews"
    max_samples: Optional[int] = 10000

    # HuggingFace/Wikipedia settings (used when dataset_source="huggingface")
    hf_cache_dir: str = "/media/arxwn/data0_4tb/datasets/wikipedia"
    hf_wikipedia_config: str = "20231101.en"
    min_article_length: int = 500
    val_fraction: float = 0.02
    max_val_samples: int = 5000
    max_train_samples: Optional[int] = None

    # Checkpointing & Analysis (step-based for large datasets)
    checkpoint_every_steps: int = 25000
    analyze_every_steps: int = 25000


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
    """Wrap (input_ids, labels) → (input_ids, {"logits": labels}).

    GPT-2 model returns ``{"logits": ..., "last_hidden_state": ...}``,
    so Keras expects labels in dict format matching the output keys.
    """
    return dataset.map(
        lambda x, y: (x, {"logits": y}),
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
    """Load train/val from Wikipedia with random holdout split."""
    train_raw, val_raw = load_wikipedia_train_val(
        cache_dir=config.hf_cache_dir,
        config_name=config.hf_wikipedia_config,
        min_article_length=config.min_article_length,
        val_fraction=config.val_fraction,
        max_train_samples=config.max_train_samples,
        max_val_samples=config.max_val_samples,
    )

    train_dataset = preprocess_clm_dataset(
        train_raw, preprocessor,
        config.max_seq_length, config.batch_size,
    )
    val_dataset = preprocess_clm_dataset(
        val_raw, preprocessor,
        config.max_seq_length, config.batch_size,
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
    # Select loss function
    if config.loss_type == "focal":
        loss_fn = FocalCausalLMLoss(
            gamma=config.focal_gamma,
            label_smoothing=config.label_smoothing,
        )
        loss_desc = f"FocalCausalLMLoss(γ={config.focal_gamma})"
    else:
        loss_fn = MaskedCausalLMLoss(
            label_smoothing=config.label_smoothing,
        )
        loss_desc = "MaskedCausalLMLoss"

    model.compile(
        optimizer=optimizer,
        loss={"logits": loss_fn},
        metrics={"logits": ["accuracy"]},
    )
    logger.info(
        f"Compiled: AdamW, peak_lr={config.learning_rate}, "
        f"wd={config.weight_decay}, loss={loss_desc}"
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

    # Steps per epoch (estimated for LR schedule)
    if config.max_train_samples:
        steps_per_epoch = config.max_train_samples // config.batch_size
    else:
        steps_per_epoch = 5000  # estimate for full Wikipedia

    # Model
    model = create_gpt2_model(config)
    compile_model(model, config, steps_per_epoch)

    # Callbacks
    callbacks, results_dir = create_nlp_callbacks(
        model_name=f"GPT2-{config.gpt2_variant}",
        results_dir_prefix="gpt2_pretrain",
        include_analyzer=False,  # Use step-based analysis instead
    )

    # Step-based checkpointing + analysis (for large datasets)
    callbacks.append(StepCheckpointCallback(
        save_dir=results_dir,
        save_every_steps=config.checkpoint_every_steps,
        analyze_every_steps=config.analyze_every_steps,
        model_name=f"GPT2-{config.gpt2_variant}",
    ))

    # Train
    logger.info(
        f"Starting training: source={config.dataset_source}, "
        f"steps_per_epoch={steps_per_epoch}"
    )
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

    # HuggingFace/Wikipedia settings
    parser.add_argument(
        "--hf-cache-dir",
        type=str,
        default="/media/arxwn/data0_4tb/datasets/wikipedia",
        help="Local cache directory for HuggingFace datasets",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Limit training articles (None=all)",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.02,
        help="Fraction of articles for validation holdout (default: 0.02)",
    )

    # Loss function
    parser.add_argument(
        "--loss-type",
        type=str,
        default="ce",
        choices=["focal", "ce"],
        help="Loss: 'ce' (default) or 'focal' (focal cross-entropy)",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=1.0,
        help="Focal loss focusing parameter gamma (default: 1.0)",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing factor (default: 0.0)",
    )

    # Checkpointing & Analysis
    parser.add_argument(
        "--checkpoint-every-steps",
        type=int,
        default=25000,
        help="Save checkpoint every N steps (default: 25000)",
    )
    parser.add_argument(
        "--analyze-every-steps",
        type=int,
        default=25000,
        help="Run model analysis every N steps (default: 25000, 0=disable)",
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
    config.hf_cache_dir = args.hf_cache_dir
    config.val_fraction = args.val_fraction
    config.checkpoint_every_steps = args.checkpoint_every_steps
    config.analyze_every_steps = args.analyze_every_steps
    config.loss_type = args.loss_type
    config.focal_gamma = args.focal_gamma
    config.label_smoothing = args.label_smoothing

    if args.max_samples is not None:
        config.max_samples = args.max_samples
    if args.max_train_samples is not None:
        config.max_train_samples = args.max_train_samples

    # TFDS settings
    config.dataset_name = args.dataset_name

    logger.info(
        f"Config: variant={config.gpt2_variant}, epochs={config.num_epochs}, "
        f"batch_size={config.batch_size}, lr={config.learning_rate}, "
        f"source={config.dataset_source}"
    )

    model, history = train_gpt2(config)
    logger.info(
        f"Pre-training complete! Model: {config.save_dir}/gpt2_final_best.keras"
    )


if __name__ == "__main__":
    main()
