"""WaveFieldLLM Pre-training Script with Causal Language Modeling.

Pre-trains a WaveFieldLLM decoder on a text dataset using next-token
prediction (causal LM). Mirrors :mod:`train.gpt2.pretrain` so the only
training-side difference between GPT-2 and WaveFieldLLM is the model class
and an optional ``--field-size`` hyperparameter.

Usage::

    # TFDS smoke run on GPU 1
    python -m train.wave_field.pretrain --gpu 1 --variant tiny \
        --dataset-source tfds --dataset-name imdb_reviews --max-samples 64 \
        --epochs 1 --batch-size 2 --max-seq-length 32

    # Wikipedia full pre-training (GPU 0)
    python -m train.wave_field.pretrain --gpu 0 --variant small --epochs 3

    # Resume from checkpoint
    python -m train.wave_field.pretrain --resume results/.../checkpoints/step_0050000.keras
"""

import os
import glob
import argparse
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, List

import keras
import numpy as np
import tensorflow as tf

from train.common import setup_gpu, set_seeds
from train.common import StepCheckpointCallback, GenerationProbeCallback
from train.common.evaluation import generate_training_curves
from train.common.nlp import (
    create_tokenizer,
    create_warmup_lr_schedule,
    create_nlp_callbacks,
    build_clm_metrics,
    prepare_dict_keyed_compile,
    augment_probe_results,
)
from train.common.clm_pretrain import (
    extract_step_from_checkpoint,
    create_clm_loss_fn,
    load_train_val_datasets,
    load_tfds_clm_datasets,
    load_hf_clm_datasets,
    make_clm_steps_per_epoch,
)
from dl_techniques.models.wave_field.model import (
    WaveFieldLLM,
    WaveFieldDecoderBlock,
)
from dl_techniques.layers.attention.wave_field_attention import (
    WaveFieldAttention,
)
from dl_techniques.initializers.identity_plus_noise import (
    IdentityPlusNoise,
)
from dl_techniques.utils.logger import logger
from dl_techniques.losses import MaskedCausalLMLoss, FocalCausalLMLoss

# DECISION plan-2026-08-12T123743-e798a9e1/D-011
# Backwards-compatible aliases for the private spellings these six functions had
# while they lived in this module -- plain assignments, so they are the SAME
# objects and not copies (the general rule is decisions.md D-010).
# WHAT NOT TO DO: do NOT delete `_extract_step_from_checkpoint` on the grounds
# that this file no longer uses it under that name. `train.wave_field.train_memory`
# imports it FROM HERE (`from train.wave_field.pretrain import
# _extract_step_from_checkpoint`) -- a cross-module coupling that predates this
# consolidation. train_memory.py has since been repointed at
# train.common.clm_pretrain directly, so this alias is now belt-and-braces for
# any importer written against the old path; it is still not dead code, and it is
# the only reason `train.wave_field.pretrain` may not be reduced to its
# WaveField-specific surface.
# See decisions.md D-011 (and D-010 for the alias-block rule it specializes).
_extract_step_from_checkpoint = extract_step_from_checkpoint
create_loss_fn = create_clm_loss_fn
_load_tfds_datasets = load_tfds_clm_datasets
_load_hf_datasets = load_hf_clm_datasets
_make_steps_per_epoch = make_clm_steps_per_epoch


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Configuration for WaveFieldLLM CLM pre-training."""

    # Model
    model_variant: str = "small"
    vocab_size: int = 50261
    max_seq_length: int = 512
    num_layers: Optional[int] = None
    num_heads: Optional[int] = None
    field_size: Optional[int] = None  # None -> 2 * max_seq_length per variant
    dropout_rate: float = 0.0
    attention_dropout_rate: float = 0.0
    tie_word_embeddings: bool = True

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
    save_dir: str = "results/wave_field_llm_pretrain"

    # Data source: "huggingface" or "tfds"
    dataset_source: str = "huggingface"

    # TFDS settings
    dataset_name: str = "imdb_reviews"
    max_samples: Optional[int] = 10000

    # HuggingFace / Wikipedia settings
    hf_cache_dir: str = "/media/arxwn/data0_4tb/datasets/wikipedia"
    hf_wikipedia_config: str = "20231101.en"
    min_article_length: int = 0
    val_fraction: float = 0.02
    max_val_samples: int = 5000
    max_train_samples: Optional[int] = None
    shuffle_shards: int = 4

    # Checkpointing & analysis (step-based for large datasets)
    checkpoint_every_steps: int = 25000
    analyze_every_steps: int = 50000
    max_checkpoints: int = 3
    steps_per_epoch: Optional[int] = None

    # Resume from checkpoint
    resume_from: Optional[str] = None
    seed: int = 42

    # Generation probes (run before each checkpoint)
    probe_prompts: List[str] = field(default_factory=lambda: [
        "The United States of America is a",
        "In mathematics, a prime number is",
        "Albert Einstein was born in",
    ])
    probe_max_tokens: int = 100
    probe_temperature: float = 0.85
    probe_top_p: float = 0.92
    probe_repetition_penalty: float = 1.3


# ---------------------------------------------------------------------
# Model creation & resume
# ---------------------------------------------------------------------


def load_model_from_checkpoint(path: str) -> Tuple[WaveFieldLLM, int]:
    """Load a WaveFieldLLM model from a ``.keras`` checkpoint."""
    logger.info(f"Resuming from checkpoint: {path}")
    model = keras.models.load_model(
        path,
        # All six classes auto-register via @register_keras_serializable;
        # listing them defensively protects against import-order surprises.
        # Keys are the Keras REGISTERED names (`"Custom>WaveFieldLLM"`, ...),
        # derived rather than hard-coded -- see D-014 in the memory_bank
        # sibling `memory_llm_custom_objects()` for why a bare class-name key
        # can never match.
        custom_objects={
            keras.saving.get_registered_name(cls): cls
            for cls in (
                MaskedCausalLMLoss,
                FocalCausalLMLoss,
                WaveFieldLLM,
                WaveFieldDecoderBlock,
                WaveFieldAttention,
                IdentityPlusNoise,
            )
        },
    )
    step = _extract_step_from_checkpoint(path)
    logger.info(
        f"Loaded model: {model.count_params():,} params, "
        f"resumed at step {step:,}"
    )
    return model, step


def create_wave_field_llm_model(config: TrainingConfig) -> WaveFieldLLM:
    """Create and build a WaveFieldLLM model from the training configuration."""
    logger.info(f"Creating WaveFieldLLM-{config.model_variant.upper()}...")

    variant_kwargs = dict(
        vocab_size=config.vocab_size,
        max_seq_len=config.max_seq_length,
        dropout_rate=config.dropout_rate,
        attention_dropout_rate=config.attention_dropout_rate,
        tie_word_embeddings=config.tie_word_embeddings,
    )
    if config.num_layers is not None:
        variant_kwargs["depth"] = config.num_layers
    if config.num_heads is not None:
        variant_kwargs["num_heads"] = config.num_heads
    if config.field_size is not None:
        variant_kwargs["field_size"] = config.field_size

    model = WaveFieldLLM.from_variant(config.model_variant, **variant_kwargs)

    # Build with a dummy forward pass to initialize weights.
    dummy = np.random.randint(
        0, config.vocab_size,
        size=(1, max(1, config.max_seq_length - 1)),
    ).astype("int32")
    model(dummy, training=False)

    logger.info(f"WaveFieldLLM model: {model.count_params():,} parameters")
    return model


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------


def compile_model(
    model: WaveFieldLLM,
    config: TrainingConfig,
    steps_per_epoch: int,
) -> None:
    lr_schedule = create_warmup_lr_schedule(
        config.learning_rate,
        config.num_epochs,
        steps_per_epoch,
        config.warmup_ratio,
    )
    prepare_dict_keyed_compile(model)
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=config.weight_decay,
            clipnorm=1.0,
        ),
        loss={"logits": create_loss_fn(config)},
        metrics={"logits": build_clm_metrics(config.encoding_name)},
    )
    logger.info(
        f"Compiled: AdamW, peak_lr={config.learning_rate}, "
        f"wd={config.weight_decay}"
    )


def train_wave_field_llm(
    config: TrainingConfig,
    model_factory: Callable[[TrainingConfig], WaveFieldLLM] = create_wave_field_llm_model,
) -> Tuple[WaveFieldLLM, keras.callbacks.History]:
    """Run WaveFieldLLM CLM pre-training."""
    logger.info("=" * 60)
    logger.info("WaveFieldLLM Causal LM Pre-training")
    logger.info("=" * 60)

    set_seeds(config.seed)
    os.makedirs(config.save_dir, exist_ok=True)

    preprocessor = create_tokenizer(
        config.encoding_name,
        config.max_seq_length,
        config.cls_token_id,
        config.sep_token_id,
        config.pad_token_id,
        config.mask_token_id,
    )

    initial_step = (
        _extract_step_from_checkpoint(config.resume_from)
        if config.resume_from else 0
    )
    data_seed = config.seed + initial_step

    train_dataset, val_dataset, n_train_articles = load_train_val_datasets(
        config, preprocessor, data_seed=data_seed,
    )

    steps_per_epoch = _make_steps_per_epoch(config, n_train_articles)

    if config.resume_from:
        model, initial_step = load_model_from_checkpoint(config.resume_from)
    else:
        model = model_factory(config)

    compile_model(model, config, steps_per_epoch)

    callbacks, results_dir = create_nlp_callbacks(
        model_name=f"WaveFieldLLM-{config.model_variant}",
        results_dir_prefix="wave_field_llm_pretrain",
        include_analyzer=False,
    )
    callbacks.append(StepCheckpointCallback(
        save_dir=results_dir,
        save_every_steps=config.checkpoint_every_steps,
        analyze_every_steps=config.analyze_every_steps,
        max_checkpoints=config.max_checkpoints,
        model_name=f"WaveFieldLLM-{config.model_variant}",
        initial_step=initial_step,
    ))

    # Generation probe context window: model's max - 1 keeps room for the
    # next token. For the smoke variant (max_seq_len=32) this means 31.
    # Common GenerationProbeCallback owns suppression/sampling/decode; the
    # closure supplies ONLY the next-position logits vector from the unpadded
    # ctx (variable-length, no padding; dict output keyed "logits"; divide-mode
    # rep penalty). Copy B's old `context_window=` maps to `ctx_length=`.
    probe_ctx = max(1, config.max_seq_length - 1)
    probe_cb = GenerationProbeCallback(
        logits_fn=lambda ctx: model(ctx, training=False)["logits"][0, -1, :].numpy(),
        repetition_penalty_mode="divide",
        ctx_length=probe_ctx,
        probe_every_steps=config.checkpoint_every_steps,
        prompts=config.probe_prompts,
        encoding_name=config.encoding_name,
        max_tokens=config.probe_max_tokens,
        temperature=config.probe_temperature,
        top_p=config.probe_top_p,
        repetition_penalty=config.probe_repetition_penalty,
        save_dir=results_dir,
        initial_step=initial_step,
    )
    probe_cb._post_generate_hook = augment_probe_results
    callbacks.append(probe_cb)

    logger.info(
        f"Starting training: source={config.dataset_source}, "
        f"steps_per_epoch~{steps_per_epoch:,}, "
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

    generate_training_curves(history, results_dir)

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
    p = argparse.ArgumentParser(
        description="WaveFieldLLM Causal LM Pre-training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Hardware
    p.add_argument("--gpu", type=int, default=None, help="GPU device index")

    # Model
    p.add_argument(
        "--variant", type=str, default="small",
        choices=list(WaveFieldLLM.MODEL_VARIANTS.keys()),
        help="WaveFieldLLM model variant",
    )
    p.add_argument("--num-layers", type=int, default=None,
                    help="Override number of decoder blocks")
    p.add_argument("--num-heads", type=int, default=None,
                    help="Override number of attention heads")
    p.add_argument(
        "--field-size", type=int, default=None,
        help="Override wave field grid resolution (default: variant-defined "
             "or 2 * max_seq_length).",
    )
    p.add_argument(
        "--tie-word-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Tie LM head to token embeddings",
    )

    # Training
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=3e-4)

    # Loss
    p.add_argument(
        "--loss-type", type=str, default="ce",
        choices=["ce", "focal"],
    )
    p.add_argument("--focal-gamma", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.0)

    # Data source
    p.add_argument(
        "--dataset-source", type=str, default="huggingface",
        choices=["tfds", "huggingface"],
    )
    p.add_argument("--dataset-name", type=str, default="imdb_reviews")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--hf-cache-dir", type=str,
                    default="/media/arxwn/data0_4tb/datasets/wikipedia")
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--val-fraction", type=float, default=0.02)
    p.add_argument(
        "--min-article-length", type=int, default=0,
        help="HF Wikipedia char-length filter. 0 = no filter (recommended "
             "for packed CLM).",
    )
    p.add_argument(
        "--shuffle-shards", type=int, default=4,
        help="HF Wikipedia parallel tokenization shards. 1 = single-thread, "
             "deterministic.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Global seed. On --resume, data seed is shifted by initial_step.",
    )

    # Checkpointing
    p.add_argument("--checkpoint-every-steps", type=int, default=25000)
    p.add_argument("--analyze-every-steps", type=int, default=50000,
                    help="0 to disable")
    p.add_argument("--max-checkpoints", type=int, default=3)
    p.add_argument(
        "--steps-per-epoch", type=int, default=None,
        help="Override LR-schedule horizon.",
    )

    # Resume
    p.add_argument("--resume", type=str, default=None,
                    help="Path to .keras checkpoint to resume from")

    # Output
    p.add_argument("--save-dir", type=str,
                    default="results/wave_field_llm_pretrain")

    return p


def _config_from_args(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        model_variant=args.variant,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        field_size=args.field_size,
        tie_word_embeddings=args.tie_word_embeddings,
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
        min_article_length=args.min_article_length,
        shuffle_shards=args.shuffle_shards,
        seed=args.seed,
        steps_per_epoch=args.steps_per_epoch,
        checkpoint_every_steps=args.checkpoint_every_steps,
        analyze_every_steps=args.analyze_every_steps,
        max_checkpoints=args.max_checkpoints,
        resume_from=args.resume,
        save_dir=args.save_dir,
    )


def main() -> None:
    args = _build_parser().parse_args()
    setup_gpu(gpu_id=args.gpu)

    config = _config_from_args(args)
    logger.info(
        f"Config: variant={config.model_variant}, "
        f"epochs={config.num_epochs}, batch={config.batch_size}, "
        f"lr={config.learning_rate}, loss={config.loss_type}, "
        f"source={config.dataset_source}, "
        f"field_size={config.field_size}"
    )

    train_wave_field_llm(config)


if __name__ == "__main__":
    main()
