"""GPT-2 Pre-training Script with Causal Language Modeling.

Pre-trains a GPT-2 decoder on a text dataset using next-token prediction
(causal LM). Supports both TFDS datasets (IMDB) and HuggingFace datasets
(Wikipedia, OpenWebText, etc.). Saves the trained model for downstream
fine-tuning or text generation.

Usage::

    # Wikipedia (default) on GPU 0
    python -m train.gpt2.pretrain --gpu 0 --variant small --epochs 3

    # TFDS dataset
    python -m train.gpt2.pretrain --dataset-source tfds --dataset-name imdb_reviews

    # Focal loss with custom gamma
    python -m train.gpt2.pretrain --loss-type focal --focal-gamma 1.0

    # Resume from checkpoint
    python -m train.gpt2.pretrain --resume results/.../checkpoints/step_0450000.keras
"""

import os
import glob
import argparse
from dataclasses import dataclass, field
from typing import Optional, Tuple

import keras
import numpy as np
import tensorflow as tf

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
# Configuration
# ---------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Configuration for GPT-2 CLM pre-training.

    All fields have sensible defaults for Wikipedia pre-training
    with a GPT-2 small model on a single GPU.
    """

    # Model
    gpt2_variant: str = "small"
    vocab_size: int = 50261
    max_seq_length: int = 512
    num_layers: Optional[int] = None
    num_heads: Optional[int] = None
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1

    # Tokenizer (Tiktoken gpt2 encoding — 50,257 base + 4 special)
    encoding_name: str = "gpt2"
    cls_token_id: int = 50257
    sep_token_id: int = 50258
    pad_token_id: int = 50259
    mask_token_id: int = 50260

    # Training
    batch_size: int = 8
    num_epochs: int = 3
    learning_rate: float = 3e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

    # Loss: "ce" (default) or "focal"
    loss_type: str = "ce"
    focal_gamma: float = 1.0
    label_smoothing: float = 0.0

    # Paths
    save_dir: str = "results/gpt2_pretrain"

    # Data source: "huggingface" or "tfds"
    dataset_source: str = "huggingface"

    # TFDS settings
    dataset_name: str = "imdb_reviews"
    max_samples: Optional[int] = 10000

    # HuggingFace / Wikipedia settings
    hf_cache_dir: str = "/media/arxwn/data0_4tb/datasets/wikipedia"
    hf_wikipedia_config: str = "20231101.en"
    min_article_length: int = 500
    val_fraction: float = 0.02
    max_val_samples: int = 5000
    max_train_samples: Optional[int] = None

    # Checkpointing & analysis (step-based for large datasets)
    checkpoint_every_steps: int = 25000
    analyze_every_steps: int = 50000
    max_checkpoints: int = 3

    # Resume from checkpoint
    resume_from: Optional[str] = None


# ---------------------------------------------------------------------
# Step-based Checkpoint & Analysis Callback
# ---------------------------------------------------------------------


class StepCheckpointCallback(keras.callbacks.Callback):
    """Save model checkpoints and run weight/spectral analysis at
    fixed step intervals.

    For large datasets where one epoch = 300K+ steps, epoch-level
    checkpoints are too infrequent. Keeps only the N most recent
    checkpoints to limit disk usage.

    :param save_dir: Root directory (creates ``checkpoints/`` and
        ``step_analysis/`` subdirectories).
    :param save_every_steps: Checkpoint interval in training steps.
    :param analyze_every_steps: Analysis interval. 0 to disable.
    :param max_checkpoints: Keep only the N most recent checkpoints.
    :param model_name: Label for analyzer output.
    """

    def __init__(
        self,
        save_dir: str,
        save_every_steps: int = 25000,
        analyze_every_steps: int = 50000,
        max_checkpoints: int = 3,
        model_name: str = "gpt2",
        initial_step: int = 0,
    ):
        super().__init__()
        self.save_every_steps = save_every_steps
        self.analyze_every_steps = analyze_every_steps
        self.max_checkpoints = max_checkpoints
        self.model_name = model_name
        self._global_step = initial_step

        self._ckpt_dir = os.path.join(save_dir, "checkpoints")
        self._analysis_dir = os.path.join(save_dir, "step_analysis")
        os.makedirs(self._ckpt_dir, exist_ok=True)
        if analyze_every_steps > 0:
            os.makedirs(self._analysis_dir, exist_ok=True)

        # Spectral analysis (WeightWatcher SVD) OOMs on large models
        # (124M+) due to cumulative memory pressure during long training
        # runs. Only enable weight distribution analysis by default.
        self._analysis_config = AnalysisConfig(
            analyze_weights=True,
            analyze_spectral=False,
            analyze_calibration=False,
            analyze_information_flow=False,
            analyze_training_dynamics=False,
            verbose=False,
        )
        logger.info(
            f"StepCheckpointCallback: save every {save_every_steps} steps, "
            f"analyze every {analyze_every_steps} steps, "
            f"keep max {max_checkpoints} checkpoints"
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

    def on_train_end(self, logs=None):
        path = os.path.join(self._ckpt_dir, "final.keras")
        self.model.save(path)
        logger.info(f"Final checkpoint saved: {path}")

    def _save_checkpoint(self):
        path = os.path.join(
            self._ckpt_dir, f"step_{self._global_step:07d}.keras"
        )
        self.model.save(path)
        logger.info(f"Checkpoint saved: {path} (step {self._global_step:,})")
        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        """Remove oldest checkpoints beyond ``max_checkpoints``."""
        ckpts = sorted(glob.glob(
            os.path.join(self._ckpt_dir, "step_*.keras")
        ))
        while len(ckpts) > self.max_checkpoints:
            old = ckpts.pop(0)
            os.remove(old)
            logger.info(f"Removed old checkpoint: {old}")

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
            analyzer.analyze()
            logger.info(f"Step analysis complete: step {self._global_step:,}")
        except Exception as e:
            logger.error(
                f"Step analysis failed at step {self._global_step}: {e}"
            )


# ---------------------------------------------------------------------
# Model Creation & Resume
# ---------------------------------------------------------------------


def _extract_step_from_checkpoint(path: str) -> int:
    """Extract the training step from a checkpoint filename.

    Handles ``step_0025000.keras`` and ``final.keras`` patterns.
    Returns 0 if the step cannot be determined.
    """
    import re
    basename = os.path.basename(path)
    match = re.search(r"step_(\d+)", basename)
    if match:
        return int(match.group(1))
    return 0


def load_model_from_checkpoint(
    path: str,
) -> Tuple[GPT2, int]:
    """Load a GPT-2 model from a ``.keras`` checkpoint.

    :param path: Path to the checkpoint file.
    :return: ``(model, step)`` — the loaded model and the training step
        extracted from the filename (0 if unknown).
    """
    logger.info(f"Resuming from checkpoint: {path}")
    model = keras.models.load_model(
        path,
        custom_objects={
            "MaskedCausalLMLoss": MaskedCausalLMLoss,
            "FocalCausalLMLoss": FocalCausalLMLoss,
        },
    )
    step = _extract_step_from_checkpoint(path)
    logger.info(
        f"Loaded model: {model.count_params():,} params, "
        f"resumed at step {step:,}"
    )
    return model, step


def create_gpt2_model(config: TrainingConfig) -> GPT2:
    """Create and build a GPT-2 model from the training configuration."""
    logger.info(f"Creating GPT-2-{config.gpt2_variant.upper()}...")

    # Build variant kwargs, only overriding if explicitly set
    variant_kwargs = dict(
        vocab_size=config.vocab_size,
        max_seq_len=config.max_seq_length,
        dropout_rate=config.dropout_rate,
        attention_dropout_rate=config.attention_dropout_rate,
    )
    if config.num_layers is not None:
        variant_kwargs["depth"] = config.num_layers
    if config.num_heads is not None:
        variant_kwargs["num_heads"] = config.num_heads

    model = GPT2.from_variant(config.gpt2_variant, **variant_kwargs)

    # Build with a dummy forward pass to initialize weights
    dummy = np.random.randint(
        0, config.vocab_size,
        size=(1, config.max_seq_length - 1),
    ).astype("int32")
    model(dummy, training=False)

    logger.info(f"GPT-2 model: {model.count_params():,} parameters")
    return model


# ---------------------------------------------------------------------
# Loss Construction
# ---------------------------------------------------------------------


def create_loss_fn(config: TrainingConfig) -> keras.losses.Loss:
    """Create the CLM loss function from configuration."""
    if config.loss_type == "focal":
        loss_fn = FocalCausalLMLoss(
            gamma=config.focal_gamma,
            label_smoothing=config.label_smoothing,
        )
        logger.info(f"Loss: FocalCausalLMLoss(γ={config.focal_gamma})")
    else:
        loss_fn = MaskedCausalLMLoss(
            label_smoothing=config.label_smoothing,
        )
        logger.info("Loss: MaskedCausalLMLoss")
    return loss_fn


# ---------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------


def load_train_val_datasets(
    config: TrainingConfig,
    preprocessor,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Load, preprocess, and wrap train/val datasets for the dict-output model."""
    if config.dataset_source == "tfds":
        train_ds, val_ds = _load_tfds_datasets(config, preprocessor)
    elif config.dataset_source == "huggingface":
        train_ds, val_ds = _load_hf_datasets(config, preprocessor)
    else:
        raise ValueError(
            f"Unknown dataset_source: {config.dataset_source!r}. "
            f"Use 'tfds' or 'huggingface'."
        )

    # Wrap labels for dict-output model: (x, y) → (x, {"logits": y})
    wrap = lambda ds: ds.map(
        lambda x, y: (x, {"logits": y}),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return wrap(train_ds), wrap(val_ds)


def _load_tfds_datasets(config, preprocessor):
    """Load train/val from TFDS (e.g. IMDB)."""
    train = preprocess_clm_dataset(
        load_text_dataset(config.dataset_name, "train", config.max_samples),
        preprocessor, config.max_seq_length, config.batch_size,
    )
    val = preprocess_clm_dataset(
        load_text_dataset(config.dataset_name, "test", config.max_samples),
        preprocessor, config.max_seq_length, config.batch_size,
    )
    return train, val


def _load_hf_datasets(config, preprocessor):
    """Load train/val from Wikipedia with random holdout split."""
    train_raw, val_raw = load_wikipedia_train_val(
        cache_dir=config.hf_cache_dir,
        config_name=config.hf_wikipedia_config,
        min_article_length=config.min_article_length,
        val_fraction=config.val_fraction,
        max_train_samples=config.max_train_samples,
        max_val_samples=config.max_val_samples,
    )
    train = preprocess_clm_dataset(
        train_raw, preprocessor,
        config.max_seq_length, config.batch_size,
    )
    val = preprocess_clm_dataset(
        val_raw, preprocessor,
        config.max_seq_length, config.batch_size,
    )
    return train, val


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------


def compile_model(
    model: GPT2,
    config: TrainingConfig,
    steps_per_epoch: int,
) -> None:
    """Compile GPT-2 with AdamW, warmup + cosine decay, and CLM loss."""
    lr_schedule = create_warmup_lr_schedule(
        config.learning_rate,
        config.num_epochs,
        steps_per_epoch,
        config.warmup_ratio,
    )
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=config.weight_decay,
            clipnorm=1.0,
        ),
        loss={"logits": create_loss_fn(config)},
        metrics={"logits": ["accuracy"]},
    )
    logger.info(
        f"Compiled: AdamW, peak_lr={config.learning_rate}, "
        f"wd={config.weight_decay}"
    )


def _estimate_steps_per_epoch(config: TrainingConfig) -> int:
    """Estimate steps per epoch for the LR schedule."""
    if config.max_train_samples:
        return config.max_train_samples // config.batch_size
    # Full Wikipedia (~4.85M articles after filtering) / batch_size
    return 4_850_000 // config.batch_size


def train_gpt2(
    config: TrainingConfig,
) -> Tuple[GPT2, keras.callbacks.History]:
    """Run GPT-2 CLM pre-training.

    :param config: Training configuration.
    :return: Trained model and training history.
    """
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
    train_dataset, val_dataset = load_train_val_datasets(
        config, preprocessor,
    )

    # Model — resume from checkpoint or create fresh
    steps_per_epoch = _estimate_steps_per_epoch(config)
    initial_step = 0

    if config.resume_from:
        model, initial_step = load_model_from_checkpoint(config.resume_from)
    else:
        model = create_gpt2_model(config)

    compile_model(model, config, steps_per_epoch)

    # Callbacks: standard NLP callbacks + step-based checkpointing
    callbacks, results_dir = create_nlp_callbacks(
        model_name=f"GPT2-{config.gpt2_variant}",
        results_dir_prefix="gpt2_pretrain",
        include_analyzer=False,
    )
    callbacks.append(StepCheckpointCallback(
        save_dir=results_dir,
        save_every_steps=config.checkpoint_every_steps,
        analyze_every_steps=config.analyze_every_steps,
        max_checkpoints=config.max_checkpoints,
        model_name=f"GPT2-{config.gpt2_variant}",
        initial_step=initial_step,
    ))

    # Train
    logger.info(
        f"Starting training: source={config.dataset_source}, "
        f"steps_per_epoch≈{steps_per_epoch:,}, "
        f"batch_size={config.batch_size}"
    )
    history = model.fit(
        train_dataset,
        epochs=config.num_epochs,
        callbacks=callbacks,
        validation_data=val_dataset,
        verbose=1,
    )
    logger.info("Training completed!")

    # Summary
    if "val_loss" in history.history:
        best_epoch = tf.argmin(history.history["val_loss"]).numpy()
        logger.info(
            f"Best epoch: {best_epoch + 1} "
            f"(val_loss: {history.history['val_loss'][best_epoch]:.4f})"
        )

    return model, history


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all training options."""
    p = argparse.ArgumentParser(
        description="GPT-2 Causal LM Pre-training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Hardware
    p.add_argument("--gpu", type=int, default=None, help="GPU device index")

    # Model
    p.add_argument(
        "--variant", type=str, default="small",
        choices=list(GPT2.MODEL_VARIANTS.keys()),
        help="GPT-2 model variant",
    )
    p.add_argument("--num-layers", type=int, default=None,
                    help="Override number of transformer layers")
    p.add_argument("--num-heads", type=int, default=None,
                    help="Override number of attention heads")

    # Training
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=3e-4)

    # Loss
    p.add_argument(
        "--loss-type", type=str, default="ce",
        choices=["ce", "focal"],
        help="'ce' (MaskedCausalLMLoss) or 'focal' (FocalCausalLMLoss)",
    )
    p.add_argument("--focal-gamma", type=float, default=1.0,
                    help="Focal loss gamma (only if --loss-type focal)")
    p.add_argument("--label-smoothing", type=float, default=0.0)

    # Data source
    p.add_argument(
        "--dataset-source", type=str, default="huggingface",
        choices=["tfds", "huggingface"],
    )
    p.add_argument("--dataset-name", type=str, default="imdb_reviews",
                    help="TFDS dataset name")
    p.add_argument("--max-samples", type=int, default=None,
                    help="TFDS max samples")
    p.add_argument("--hf-cache-dir", type=str,
                    default="/media/arxwn/data0_4tb/datasets/wikipedia")
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--val-fraction", type=float, default=0.02)

    # Checkpointing
    p.add_argument("--checkpoint-every-steps", type=int, default=25000)
    p.add_argument("--analyze-every-steps", type=int, default=50000,
                    help="0 to disable")
    p.add_argument("--max-checkpoints", type=int, default=3)

    # Resume
    p.add_argument(
        "--resume", type=str, default=None,
        help="Path to .keras checkpoint to resume training from",
    )

    # Output
    p.add_argument("--save-dir", type=str, default="results/gpt2_pretrain")

    return p


def _config_from_args(args: argparse.Namespace) -> TrainingConfig:
    """Map parsed CLI args to a TrainingConfig."""
    return TrainingConfig(
        gpt2_variant=args.variant,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        learning_rate=args.learning_rate,
        loss_type=args.loss_type,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        dataset_source=args.dataset_source,
        dataset_name=args.dataset_name,
        max_samples=args.max_samples,
        hf_cache_dir=args.hf_cache_dir,
        max_train_samples=args.max_train_samples,
        val_fraction=args.val_fraction,
        checkpoint_every_steps=args.checkpoint_every_steps,
        analyze_every_steps=args.analyze_every_steps,
        max_checkpoints=args.max_checkpoints,
        resume_from=args.resume,
        save_dir=args.save_dir,
    )


def main() -> None:
    """Main entry point for GPT-2 CLM pre-training."""
    args = _build_parser().parse_args()
    setup_gpu(gpu_id=args.gpu)

    config = _config_from_args(args)
    logger.info(
        f"Config: variant={config.gpt2_variant}, "
        f"epochs={config.num_epochs}, batch={config.batch_size}, "
        f"lr={config.learning_rate}, loss={config.loss_type}, "
        f"source={config.dataset_source}"
    )

    train_gpt2(config)


if __name__ == "__main__":
    main()
