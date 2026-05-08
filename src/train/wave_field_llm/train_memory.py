"""WaveFieldMemoryLLM Training Script (Phase 1-3 curriculum CLM).

Mirrors :mod:`train.wave_field_llm.pretrain` 1:1 except for:

- Model factory: :class:`WaveFieldMemoryLLM` (`from_variant`).
- Compile: two AdamW optimizers (backbone vs memory), passed via the
  model's custom :meth:`compile` override.
- Callbacks: :class:`PhaseScheduler` is appended (with a warmup dataset
  derived from the first 64 train batches) so trainable flags + aux
  losses + KMeans warmup advance with global step.
- New CLI flags: ``--init-from``, ``--phase1-steps``, ``--phase2-steps``,
  ``--phase3-steps``, ``--memory-lr``, ``--top-k``.

The model returns a dict ``{"logits", "last_hidden_state"}`` keyed on
``"logits"`` so :func:`prepare_dict_keyed_compile` + ``MaskedCausalLMLoss``
+ ``build_clm_metrics`` work unchanged.

Usage::

    # Smoke test
    MPLBACKEND=Agg .venv/bin/python -m train.wave_field_llm.train_memory \\
        --variant tiny --max-seq-length 64 --batch-size 2 --epochs 1 \\
        --steps-per-epoch 4 --phase1-steps 2 --phase2-steps 2 \\
        --phase3-steps 100 --max-train-samples 16 --val-fraction 0.5 \\
        --max-samples 16 --save-dir /tmp/mem_smoke

    # Resume from Phase-1 WaveFieldLLM checkpoint
    MPLBACKEND=Agg .venv/bin/python -m train.wave_field_llm.train_memory \\
        --gpu 0 --variant small --epochs 5 \\
        --init-from results/wave_field_llm_pretrain_*/checkpoints/final.keras
"""

import argparse
import os
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import keras
import numpy as np
import tensorflow as tf

from train.common import setup_gpu
from train.common.evaluation import generate_training_curves
from train.common.nlp import (
    create_tokenizer,
    load_text_dataset,
    preprocess_clm_dataset,
    create_warmup_lr_schedule,
    create_nlp_callbacks,
    estimate_clm_steps_per_epoch,
    build_clm_metrics,
    prepare_dict_keyed_compile,
    augment_probe_results,
)
from train.wave_field_llm.pretrain import (
    StepCheckpointCallback,
    GenerationProbeCallback,
    _extract_step_from_checkpoint,
)
from dl_techniques.models.memory_bank.wave_field_memory_llm import (
    WaveFieldMemoryLLM,
    memory_llm_custom_objects,
)
from dl_techniques.models.memory_bank.phase_scheduler import PhaseScheduler
from dl_techniques.utils.logger import logger
from dl_techniques.utils.weight_transfer import load_weights_from_checkpoint
from dl_techniques.datasets.nlp import load_wikipedia_train_val
from dl_techniques.losses import MaskedCausalLMLoss, FocalCausalLMLoss


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Training configuration for WaveFieldMemoryLLM CLM curriculum."""

    # Model
    model_variant: str = "small"
    vocab_size: int = 50261
    max_seq_length: int = 512
    field_size: Optional[int] = None
    dropout_rate: float = 0.0
    attention_dropout_rate: float = 0.0
    tie_word_embeddings: bool = True
    top_k: int = 32

    # Tokenizer
    encoding_name: str = "gpt2"
    cls_token_id: int = 50257
    sep_token_id: int = 50258
    pad_token_id: int = 50259
    mask_token_id: int = 50260

    # Training
    batch_size: int = 8
    num_epochs: int = 3
    learning_rate: float = 1e-5  # backbone LR (Phase 3)
    memory_lr: float = 3e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

    # Loss
    loss_type: str = "ce"
    focal_gamma: float = 1.0
    label_smoothing: float = 0.0

    # Phase boundaries (default 50K / 25K / 100K).
    phase1_steps: int = 50_000
    phase2_steps: int = 25_000
    phase3_steps: int = 100_000
    warmup_num_batches: int = 64

    # Resume / init.
    resume_from: Optional[str] = None
    init_from: Optional[str] = None

    # Paths
    save_dir: str = "results/wave_field_memory_llm"

    # Data source
    dataset_source: str = "huggingface"
    dataset_name: str = "imdb_reviews"
    max_samples: Optional[int] = 10_000

    hf_cache_dir: str = "/media/arxwn/data0_4tb/datasets/wikipedia"
    hf_wikipedia_config: str = "20231101.en"
    min_article_length: int = 0
    val_fraction: float = 0.02
    max_val_samples: int = 5_000
    max_train_samples: Optional[int] = None
    shuffle_shards: int = 4

    # Checkpointing
    checkpoint_every_steps: int = 25_000
    analyze_every_steps: int = 50_000
    max_checkpoints: int = 3
    steps_per_epoch: Optional[int] = None
    seed: int = 42

    # Generation probes
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
# Model creation / resume
# ---------------------------------------------------------------------


def create_memory_model(config: TrainingConfig) -> WaveFieldMemoryLLM:
    """Build a :class:`WaveFieldMemoryLLM` from the variant ladder."""
    logger.info(f"Creating WaveFieldMemoryLLM-{config.model_variant.upper()}...")

    overrides = dict(
        vocab_size=config.vocab_size,
        max_seq_len=config.max_seq_length,
        dropout_rate=config.dropout_rate,
        attention_dropout_rate=config.attention_dropout_rate,
        tie_word_embeddings=config.tie_word_embeddings,
        top_k=config.top_k,
    )
    if config.field_size is not None:
        overrides["field_size"] = config.field_size

    model = WaveFieldMemoryLLM.from_variant(config.model_variant, **overrides)

    # Build with a dummy forward pass (parity with WaveFieldLLM trainer).
    dummy = np.random.randint(
        0, config.vocab_size,
        size=(1, max(1, config.max_seq_length - 1)),
    ).astype("int32")
    model(dummy, training=False)

    n_params = sum(int(np.prod(w.shape)) for w in model.weights)
    logger.info(f"WaveFieldMemoryLLM model: {n_params:,} parameters")
    return model


def load_memory_model_from_checkpoint(
    path: str,
) -> Tuple[WaveFieldMemoryLLM, int]:
    logger.info(f"Resuming from checkpoint: {path}")
    model = keras.models.load_model(
        path, custom_objects=memory_llm_custom_objects(),
    )
    step = _extract_step_from_checkpoint(path)
    n_params = sum(int(np.prod(w.shape)) for w in model.weights)
    logger.info(f"Loaded model: {n_params:,} params, resumed at step {step:,}")
    return model, step


# ---------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------


def create_loss_fn(config: TrainingConfig) -> keras.losses.Loss:
    if config.loss_type == "focal":
        return FocalCausalLMLoss(
            gamma=config.focal_gamma,
            label_smoothing=config.label_smoothing,
        )
    return MaskedCausalLMLoss(label_smoothing=config.label_smoothing)


# ---------------------------------------------------------------------
# Data loading (mirrors pretrain.py)
# ---------------------------------------------------------------------


def _load_tfds_datasets(config, preprocessor):
    train = preprocess_clm_dataset(
        load_text_dataset(config.dataset_name, "train", config.max_samples),
        preprocessor, config.max_seq_length, config.batch_size,
    )
    val = preprocess_clm_dataset(
        load_text_dataset(config.dataset_name, "test", config.max_samples),
        preprocessor, config.max_seq_length, config.batch_size,
    )
    return train, val


def _load_hf_datasets(config, preprocessor, data_seed):
    train_raw, val_raw, n_train, _ = load_wikipedia_train_val(
        cache_dir=config.hf_cache_dir,
        config_name=config.hf_wikipedia_config,
        min_article_length=config.min_article_length,
        val_fraction=config.val_fraction,
        max_train_samples=config.max_train_samples,
        max_val_samples=config.max_val_samples,
        seed=data_seed,
        num_shards=config.shuffle_shards,
        return_counts=True,
    )
    train = preprocess_clm_dataset(
        train_raw, preprocessor, config.max_seq_length, config.batch_size,
    )
    val = preprocess_clm_dataset(
        val_raw, preprocessor, config.max_seq_length, config.batch_size,
    )
    return train, val, n_train


def load_train_val_datasets(config, preprocessor, data_seed):
    n_train_articles: Optional[int] = None
    if config.dataset_source == "tfds":
        train_ds, val_ds = _load_tfds_datasets(config, preprocessor)
    elif config.dataset_source == "huggingface":
        train_ds, val_ds, n_train_articles = _load_hf_datasets(
            config, preprocessor, data_seed,
        )
    else:
        raise ValueError(
            f"Unknown dataset_source: {config.dataset_source!r}"
        )
    wrap = lambda ds: ds.map(
        lambda x, y: (x, {"logits": y}),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return wrap(train_ds), wrap(val_ds), n_train_articles


# ---------------------------------------------------------------------
# Compile (dual optimizer)
# ---------------------------------------------------------------------


def compile_memory_model(
    model: WaveFieldMemoryLLM,
    config: TrainingConfig,
    steps_per_epoch: int,
) -> None:
    backbone_lr_schedule = create_warmup_lr_schedule(
        config.learning_rate,
        config.num_epochs,
        steps_per_epoch,
        config.warmup_ratio,
    )
    memory_lr_schedule = create_warmup_lr_schedule(
        config.memory_lr,
        config.num_epochs,
        steps_per_epoch,
        config.warmup_ratio,
    )

    backbone_opt = keras.optimizers.AdamW(
        learning_rate=backbone_lr_schedule,
        weight_decay=config.weight_decay,
        clipnorm=1.0,
    )
    memory_opt = keras.optimizers.AdamW(
        learning_rate=memory_lr_schedule,
        weight_decay=config.weight_decay,
        clipnorm=1.0,
    )

    prepare_dict_keyed_compile(model)
    model.compile(
        backbone_optimizer=backbone_opt,
        memory_optimizer=memory_opt,
        loss={"logits": create_loss_fn(config)},
        metrics={"logits": build_clm_metrics(config.encoding_name)},
    )

    logger.info(
        f"Compiled: backbone_lr={config.learning_rate}, "
        f"memory_lr={config.memory_lr}, wd={config.weight_decay}"
    )


def _make_steps_per_epoch(config, n_train_articles):
    if (
        config.dataset_source == "tfds"
        and config.max_samples
        and config.steps_per_epoch is None
    ):
        return max(1, config.max_samples // config.batch_size)
    return estimate_clm_steps_per_epoch(
        num_articles=n_train_articles or config.max_train_samples,
        max_seq_length=config.max_seq_length,
        batch_size=config.batch_size,
        override=config.steps_per_epoch,
    )


# ---------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------


def train_memory_llm(
    config: TrainingConfig,
    model_factory: Callable[[TrainingConfig], WaveFieldMemoryLLM] = create_memory_model,
) -> Tuple[WaveFieldMemoryLLM, keras.callbacks.History]:
    logger.info("=" * 60)
    logger.info("WaveFieldMemoryLLM CLM Curriculum Training")
    logger.info("=" * 60)

    tf.random.set_seed(config.seed)
    keras.utils.set_random_seed(config.seed)
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
        model, initial_step = load_memory_model_from_checkpoint(config.resume_from)
    else:
        model = model_factory(config)
        if config.init_from:
            logger.info(
                f"Loading Phase-1 backbone weights from: {config.init_from}"
            )
            load_weights_from_checkpoint(
                model, config.init_from,
                skip_prefixes=("memory_", "gate_"),
            )
            # `--init-from` skips Phase 1 entirely.
            config.phase1_steps = 0

    compile_memory_model(model, config, steps_per_epoch)

    callbacks, results_dir = create_nlp_callbacks(
        model_name=f"WaveFieldMemoryLLM-{config.model_variant}",
        results_dir_prefix="wave_field_memory_llm",
        include_analyzer=False,
    )

    # Phase scheduler (warmup dataset = first N batches of train).
    phase_cb = PhaseScheduler(
        phase1_steps=config.phase1_steps,
        phase2_steps=config.phase2_steps,
        phase3_steps=config.phase3_steps,
        warmup_dataset=train_dataset.take(config.warmup_num_batches),
        warmup_num_batches=config.warmup_num_batches,
    )
    callbacks.append(phase_cb)

    callbacks.append(StepCheckpointCallback(
        save_dir=results_dir,
        save_every_steps=config.checkpoint_every_steps,
        analyze_every_steps=config.analyze_every_steps,
        max_checkpoints=config.max_checkpoints,
        model_name=f"WaveFieldMemoryLLM-{config.model_variant}",
        initial_step=initial_step,
    ))

    probe_ctx = max(1, config.max_seq_length - 1)
    probe_cb = GenerationProbeCallback(
        probe_every_steps=config.checkpoint_every_steps,
        prompts=config.probe_prompts,
        encoding_name=config.encoding_name,
        max_tokens=config.probe_max_tokens,
        temperature=config.probe_temperature,
        top_p=config.probe_top_p,
        repetition_penalty=config.probe_repetition_penalty,
        save_dir=results_dir,
        initial_step=initial_step,
        context_window=probe_ctx,
    )
    probe_cb._post_generate_hook = augment_probe_results
    callbacks.append(probe_cb)

    logger.info(
        f"Starting training: source={config.dataset_source}, "
        f"steps_per_epoch~{steps_per_epoch:,}, batch_size={config.batch_size}, "
        f"phases=({config.phase1_steps}, {config.phase2_steps}, "
        f"{config.phase3_steps})"
    )
    history = model.fit(
        train_dataset,
        epochs=config.num_epochs,
        callbacks=callbacks,
        validation_data=val_dataset,
        verbose=1,
    )
    logger.info("Training completed.")

    generate_training_curves(history, results_dir)

    if "val_loss" in history.history:
        best_epoch = int(tf.argmin(history.history["val_loss"]).numpy())
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
        description="WaveFieldMemoryLLM CLM Curriculum Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Hardware
    p.add_argument("--gpu", type=int, default=None)

    # Model
    p.add_argument(
        "--variant", type=str, default="small",
        choices=list(WaveFieldMemoryLLM.MODEL_VARIANTS.keys()),
    )
    p.add_argument("--field-size", type=int, default=None)
    p.add_argument(
        "--tie-word-embeddings",
        action=argparse.BooleanOptionalAction, default=True,
    )
    p.add_argument(
        "--top-k", type=int, default=32,
        help="Top-K retrieval count.",
    )

    # Training
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=1e-5,
                    help="Backbone learning rate (Phase 3).")
    p.add_argument("--memory-lr", type=float, default=3e-4,
                    help="Memory components learning rate.")

    # Loss
    p.add_argument("--loss-type", type=str, default="ce",
                    choices=["ce", "focal"])
    p.add_argument("--focal-gamma", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.0)

    # Phase boundaries
    p.add_argument("--phase1-steps", type=int, default=50_000)
    p.add_argument("--phase2-steps", type=int, default=25_000)
    p.add_argument("--phase3-steps", type=int, default=100_000)

    # Init / resume
    p.add_argument(
        "--init-from", type=str, default=None,
        help="Path to a Phase-1 WaveFieldLLM .keras checkpoint. "
             "Backbone is loaded; memory_/gate_ variables skipped. "
             "Forces phase1_steps=0.",
    )
    p.add_argument("--resume", type=str, default=None,
                    help="Resume full memory model from .keras checkpoint.")

    # Data source (CLM CLI uniformity — see train/CLAUDE.md)
    p.add_argument("--dataset-source", type=str, default="huggingface",
                    choices=["tfds", "huggingface"])
    p.add_argument("--dataset-name", type=str, default="imdb_reviews")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument(
        "--hf-cache-dir", type=str,
        default="/media/arxwn/data0_4tb/datasets/wikipedia",
    )
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--val-fraction", type=float, default=0.02)
    p.add_argument(
        "--min-article-length", type=int, default=0,
        help="HF Wikipedia char-length filter. 0 = no filter.",
    )
    p.add_argument(
        "--shuffle-shards", type=int, default=4,
        help="HF Wikipedia parallel tokenization shards.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Global seed (data seed shifted by initial_step on resume).",
    )

    # Checkpointing
    p.add_argument("--checkpoint-every-steps", type=int, default=25_000)
    p.add_argument("--analyze-every-steps", type=int, default=50_000)
    p.add_argument("--max-checkpoints", type=int, default=3)
    p.add_argument(
        "--steps-per-epoch", type=int, default=None,
        help="Override LR-schedule horizon.",
    )

    # Output
    p.add_argument("--save-dir", type=str,
                    default="results/wave_field_memory_llm")

    return p


def _config_from_args(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        model_variant=args.variant,
        field_size=args.field_size,
        tie_word_embeddings=args.tie_word_embeddings,
        top_k=args.top_k,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        learning_rate=args.learning_rate,
        memory_lr=args.memory_lr,
        loss_type=args.loss_type,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        phase1_steps=args.phase1_steps,
        phase2_steps=args.phase2_steps,
        phase3_steps=args.phase3_steps,
        resume_from=args.resume,
        init_from=args.init_from,
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
        save_dir=args.save_dir,
    )


def main() -> None:
    args = _build_parser().parse_args()
    setup_gpu(gpu_id=args.gpu)

    config = _config_from_args(args)
    logger.info(
        f"Config: variant={config.model_variant}, "
        f"epochs={config.num_epochs}, batch={config.batch_size}, "
        f"backbone_lr={config.learning_rate}, memory_lr={config.memory_lr}, "
        f"phases=({config.phase1_steps}, {config.phase2_steps}, "
        f"{config.phase3_steps}), top_k={config.top_k}, "
        f"init_from={config.init_from}, source={config.dataset_source}"
    )
    train_memory_llm(config)


if __name__ == "__main__":
    main()
